from __future__ import division, print_function

import collections
from collections import OrderedDict as OD
from collections import defaultdict
import sys
import tempfile

import networkx as nx
import numpy as np
import six
import torch

from utils import in_notebook
from pytorch_utils import tensor_like, iterable


class ModuleDict(torch.nn.Module):
    r"""Holds submodules in an OrderedDict.

    ModuleList can be indexed like a regular Python collections.OrderedDict,
    but modules it contains are properly registered,
    and will be visible by all Module methods.

    The ModuleDict can be accessed by layer name,
    or by layer's index in the OrderedDict.

    Arguments:
        modules (OrderedDict or List, optional)
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.keys()[key]
        return torch.nn.Module.__getattr__(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self.values()[key]
        torch.nn.Module.__setattr__(self, key, value)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules):
        self._modules.update(modules.items())


def dotdraw(G, dest=None, fontsize=10):
    if dest is None:
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
        f.close()
        dest = f.name
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(fontsize=fontsize)
    A.layout(prog='dot')
    A.draw(dest)
    if in_notebook():
        from IPython.display import SVG
        return SVG(dest)
    else:
        return dest


class AutoDestructDict(dict):
    """
    Dictionary self-deleting elements
    after a certain number of accesses
    """
    def __init__(self, dictionary, max_uses):
        super(AutoDestructDict, self).__init__(dictionary)
        self.max_uses = dict(max_uses)

    def __repr__(self):
        return "<AutoDestructDict with keys {}>".format(self.keys())

    def __getitem__(self, idx):
        out = super(AutoDestructDict, self).__getitem__(idx)
        if idx in self.max_uses:
            self.max_uses[idx] -= 1
            if self.max_uses[idx] == 0:
                del self[idx]
        return out


class DagModule(ModuleDict):

    def __init__(self, net, in_layer='data', out_layer=None, in_channels=3,
                 compute_all=False, keep_outputs=False, verbose=1):
        """
        DagModule: Explicitely implement a dag as a pytorch module
        in_layer/out_layer can be one layer, or a list of layers (same for in_channels).
        if compute_all is True, does not prune the DAG.
        if keep_outputs is True, keep outputs of each layers. (kept in self.values)
        if keep_outputs is 'top', keep output of top layers. (kept in self.values)
        modifiable attributes:
        self.out_layer
        self.compute_all

        """
        self._initialized = False
        super(DagModule, self).__init__()
        self.verbose = verbose
        self.net = self.get_net(net) if isinstance(net, basestring) else net
        if iterable(in_layer):
            self._in_layer = in_layer
            self._repeat_input = False
        else:
            self._in_layer = (in_layer,)
            in_channels = (in_channels,)
            self._repeat_input = True
        self.out_layer = out_layer
        self.compute_all = compute_all
        self.keep_outputs = keep_outputs
        self._dag = nx.DiGraph()
        self._bottoms = {}
        self._values = {}
        self.force_sync = False   # force cuda sync. between function calls for debugging
        channels = OD()
        self.current_top = OD([(i, i) for i in self._in_layer])
        for name, chan in zip(self._in_layer, in_channels):
            channels[name] = chan
            self._dag.add_node(name)
        name_config = OD([(l.name.encode('ascii'), l) for l in self.net.layer])
        for name, layer_conf in name_config.items():
            self.log(2, "Adding {}...".format(name))
            layer, out_channels, bottom, top = _getlayer(layer_conf, channels)
            self[name] = layer
            channels[name] = out_channels
            self._dag.add_node(name)
            self._bottoms[name] = list()
            for b in bottom:
                self._dag.add_edge(self.current_top[b], name)
                self._bottoms[name].append(self.current_top[b])
            for t in top:
                self.current_top[t] = name

        if not nx.is_directed_acyclic_graph(self._dag):
            raise NotImplementedError("Prototxt does not describe a DAG.")
        self._initialized = True
        self._recompute_order()

    @property
    def values(self):
        return self._values if type(self._values) is dict else dict(self._values)

    @property
    def dag(self):
        return self._dag

    @property
    def in_layer(self):
        if self._repeat_input:
            return self._in_layer[0]
        else:
            return self._in_layer

    @property
    def _out_layer(self):
        return [self.keys()[-1] if l is None else l for l in self._out_layer_none]

    @_out_layer.setter
    def _out_layer(self, value):
        self._out_layer_none = value

    @property
    def out_layer(self):
        if self._repeat_output:
            return self._out_layer[0]
        else:
            return self._out_layer

    @out_layer.setter
    def out_layer(self, value):
        if iterable(value):
            self._out_layer = value
            self._repeat_output = False
        else:
            self._out_layer = [value]
            self._repeat_output = True
        if self._initialized:
            self._recompute_order()

    def get_param_groups(self, base_lr, base_momentum, base_wd):
        lrs = defaultdict(list) # index (lr_mult, decay_mult, momentum)
        for name, layer in self.items():
            meta = getattr(layer, "meta", {})
            for name, param in layer.named_parameters():
                lr_mult = meta.get('lr_mult', 1.0)
                momentum = meta.get('momentum', base_momentum)
                decay_mult = meta.get('decay_mult', 1.0)
                lrs[lr_mult, decay_mult, momentum].append(param)
        param_groups_mults = list(lrs.items())    # to preserve order
        groups = [{'params': params, 'lr': lr_mult * base_lr, 'wd': decay_mult * base_wd, 'm': m}
                  for (lr_mult, decay_mult, m), params in param_groups_mults]
        self.param_groups_keys = [t[0] for t in param_groups_mults]

        return groups

    def alter_param_groups(self, param_groups, new_base_lr=None, new_base_momentum=None, new_base_wd=None):
        for (lr_mult, decay_mult, m), param_group in zip(self.param_groups_keys, param_groups):
            if new_base_lr is not None:
                param_group['lr'] = lr_mult * new_base_lr
            if new_base_wd is not None:
                param_group['weight_decay'] = decay_mult * new_base_wd
            if new_base_wd is not None:
                param_group['momentum'] = m

    def _prune_DAG(self):
        """
        Prune DAG based on in_layer and out_layer
        Keep only needed nodes
        """
        needed = set(self._out_layer)
        for o in self._out_layer:
            needed = needed | nx.ancestors(self._dag, o)
        pruned = self._dag.subgraph(needed)
        return pruned

    def draw_dag(self, dest=None, pruned=True, **kwargs):
        dag = self._dag if self.compute_all or not pruned else self._prune_DAG()
        return dotdraw(dag, dest, **kwargs)

    def _recompute_order(self):
        """
        Find computation order from self._in_layer to self._out_layer.
        Avoids computing unused layers.
        Reuses buffers efficiently to allow garbage collection.
        """
        dag = self._dag if self.compute_all else self._prune_DAG()
        headless = dag.subgraph(set(dag.nodes) - set(self._in_layer))
        order = nx.topological_sort(headless)
        n_accesses = dict(dag.out_degree(set(dag.nodes) - set(self._out_layer)))
        self._order = list(order)
        self._n_accesses = n_accesses

    def _number_accesses(self):
        if self.keep_outputs is True:
            n_accesses = {}
        elif self.keep_outputs is 'top':
            n_accesses = {k: v for (k, v) in self._n_accesses.items() if k not in self.current_top.values()}
        else:
            n_accesses = self._n_accesses
        return n_accesses

    def forward(self, *x):
        """
        Forwards x from in_layer to out_layer
        """
        self._values = dict()
        for name, x in zip(self._in_layer, x):
            self._values[name] = x

        self._values = AutoDestructDict(self._values, self._number_accesses())
        for name in self._order:
            self.log(2, "{}({})...".format(name, ', '.join(self._bottoms[name])))
            if self.force_sync: torch.cuda.synchronize()
            self._values[name] = self[name](*[self._values[p] for p in self._bottoms[name]])
        out = tuple(self._values[o] for o in self._out_layer)
        return out[0] if self._repeat_output else out

    def log(self, min_verbose, *args, **kwargs):
        if self.verbose >= min_verbose:
            print(*args, **kwargs)

