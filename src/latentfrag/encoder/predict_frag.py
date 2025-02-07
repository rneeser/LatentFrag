import argparse
from pathlib import Path
import torch
from glob import glob

from rdkit import Chem

from latentfrag.encoder.models.lightning_modules import FragEmbed

parser = argparse.ArgumentParser()
parser.add_argument('--sdf_dir', type=Path, help='Input SDF directory', default=None)
parser.add_argument('--sdf_file', type=Path, help='Input SDF file with ONE molecule', default=None)
parser.add_argument('--SMILES', type=str, help='Input SMILES string', default=None)
parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
parser.add_argument('--out_dir', type=Path, required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

assert args.sdf_dir is not None or args.sdf_file is not None or args.SMILES is not None, \
    'Either --sdf_dir, --sdf_file or --SMILES must be provided'

# make dir
args.out_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = FragEmbed.load_from_checkpoint(args.checkpoint,
                                        map_location=device,
                                        strict=False)
model.to(device)
model.eval()

# Process SDF file
if args.sdf_dir is not None:
    sdf_files = glob(str(args.sdf_dir / '*.sdf'))
    sdf_files = [Path(f) for f in sdf_files]
elif args.sdf_file is not None:
    sdf_files = [args.sdf_file]
else:
    sdf_files = [Chem.MolFromSmiles(args.SMILES)]

# Predict and save
for sdf_file in sdf_files:
    ligand = model.encode_ligand(sdf_file)

    if isinstance(sdf_file, Chem.Mol):
        folder_name = f'smi_{args.SMILES}'
    else:
        folder_name = sdf_file.stem
    save_path_node = args.out_dir / f'{folder_name}_atom_emb.pt'
    save_path_global = args.out_dir / f'{folder_name}_global_emb.pt'

    node_emb = ligand['desc'].cpu()
    global_emb = ligand['desc_global'].cpu()

    torch.save(node_emb, save_path_node)
    torch.save(global_emb, save_path_global)

    print(f'Saved embeddings for {sdf_file} to {save_path_node} and {save_path_global}')

print('Done!')
