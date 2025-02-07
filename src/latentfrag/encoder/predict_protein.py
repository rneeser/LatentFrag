import argparse
from pathlib import Path
import torch

from latentfrag.encoder.models.lightning_modules import FragEmbed

parser = argparse.ArgumentParser()
parser.add_argument('--pdb_file', type=Path, help='Input PDB file', required=True)
parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
parser.add_argument('--out_dir', type=Path, required=True)
parser.add_argument('--msms_bin', type=Path, required=True, help='Path to the MSMS binary')
parser.add_argument('--msms_res', type=float, default=1.0, help='MSMS resolution')
parser.add_argument('--msms_subsampling', type=int, default=20, help='MSMS sup-sampling')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# make dir
args.out_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = FragEmbed.load_from_checkpoint(args.checkpoint,
                                        map_location=device,
                                        surface_params={'msms_bin': args.msms_bin,
                                                    'resolution': args.msms_res,
                                                    'sup_sampling': args.msms_subsampling},
                                        strict=False)
model.to(device)
model.eval()

prot_encoded = model.encode_protein(args.pdb_file, faces=True)

save_path = args.out_dir / f'{args.pdb_file.stem}_prot_encoded.pt'
torch.save(prot_encoded, save_path)

print(f'Saved protein encoding for {args.pdb_file} to {save_path}')

print('Done!')