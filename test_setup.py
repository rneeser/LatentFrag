import torch
print(f'Torch CUDA is available: {torch.cuda.is_available()}')

# To check PyG
from torch_geometric.nn import EdgeConv, Reshape
print('PyG check passed')

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
