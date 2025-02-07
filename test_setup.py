import torch
print(f'Torch CUDA is available: {torch.cuda.is_available()}')

# To check PyG
from torch_geometric.nn import EdgeConv, Reshape
print('PyG check passed')

# To check PyKeops
from pykeops.torch import LazyTensor
x = torch.rand((10, 3), device='cuda:0', dtype=torch.float32)
y = torch.rand((12, 3), device='cuda:0', dtype=torch.float32)
x_i = LazyTensor(x[:, None, :])
y_j = LazyTensor(y[None, :, :])
D_ij = ((x_i - y_j) ** 2).sum(-1)
res = D_ij.sqrt().exp().sum(0).sum(0)
print(f'PyKeOps check passed')

# Check PyKeops in multi-GPU setting
if torch.cuda.device_count() > 1:
    x = torch.rand((10, 3), device='cuda:1', dtype=torch.float32)
    y = torch.rand((12, 3), device='cuda:1', dtype=torch.float32)
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    res = D_ij.sqrt().exp().sum(0).sum(0)
    print(f'First PyKeOps multi-GPU check passed')

# To check rdkit
from rdkit import Chem
Chem.MolFromSmiles('C')
print('RDKit check passed')

# To check torch_scatter
from torch_scatter import scatter
x = torch.arange(10, device='cuda:0')
idx = torch.tensor([0] * 2 + [1] * 5 + [2] * 3, dtype=torch.long, device='cuda:0')
reduced = scatter(x, index=idx, reduce="sum")
assert len(reduced) == 3
print('Torch-scatter check passed')
