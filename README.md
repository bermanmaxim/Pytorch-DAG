# Pytorch-DAG
Pytorch nn.module with arbitrary directed acyclic graph

Explicitely implement a DAG, choose which intermediate nodes to return. 

If `compute_all` is false, only the nodes needed for outputting the target nodes will be computed (easy partial model computation).

*Possibly not upgraded for PyTorch 0.4 yet.*