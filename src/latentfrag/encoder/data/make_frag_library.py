from pathlib import Path
import argparse

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm

from latentfrag.encoder.models.lightning_modules import FragEmbed
from latentfrag.encoder.utils.data import TensorDict, featurize_ligand
from latentfrag.fm.utils.constants import FLOAT_TYPE

def get_embedding(mol, model):
    use_ev = model.hparams.get('ligand_encoder_params').get('use_ev', False)
    device = model.device
    frag_feat = featurize_ligand(mol, use_ev=use_ev)

    frag_feat = TensorDict(**frag_feat)
    frag_feat['batch'] = torch.zeros(len(frag_feat['xyz']), dtype=int)
    frag_feat = frag_feat.to(device)
    frag_feat['desc'], frag_feat['desc_global'] = model.ligand_encoder(
            frag_feat["xyz"], frag_feat["types"], frag_feat["batch"],
            frag_feat["bonds"], frag_feat["bond_types"], frag_feat["mol_feats"], return_global=True)
    return frag_feat['desc_global'].detach().cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sdf_path', type=str, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ckpt', type=str, required=True)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FragEmbed.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    model.eval()

    use_ev = model.hparams.get('ligand_encoder_params').get('use_ev', False)
    assert not use_ev, "This script does not support embeddings with exit vectors."

    frags = Chem.SDMolSupplier(args.sdf_path, sanitize=True)

    ckpt_name = Path(args.ckpt).parent.parent.name
    outdir = args.out / ckpt_name
    outdir.mkdir(parents=True, exist_ok=True)

    library = {}

    for uuid, frag in tqdm(enumerate(frags), total=len(frags), desc='Processing fragments'):
        if frag is None:
            continue
        embedding = get_embedding(frag, model)

        ff = AllChem.UFFGetMoleculeForceField(frag, confId=0)
        energy = ff.CalcEnergy()

        library[uuid] = {
            'smiles': Chem.MolToSmiles(frag),
            'embedding': embedding.to(FLOAT_TYPE),
            'energy': energy,
        }

    # save full library
    torch.save(library, outdir / 'library_all_uuid2data.pt')

    library_df = pd.DataFrame(library).T
    library_df['uuid'] = library_df.index
    library_df = library_df.astype({'energy': float})

    library_df_sorted = library_df.sort_values(by=['energy','smiles'], ascending=[True, True])
    library_df_dedup = library_df_sorted.drop_duplicates(subset=['smiles'], keep='first')
    library_df_unique = library_df_dedup.groupby('smiles').first()
    smi2options = library_df_dedup.groupby('smiles')['uuid'].apply(list)

    # save unique library
    library_unique = {}
    library_options = {}
    for smi, row in library_df_unique.iterrows():
        uuid = row['uuid']
        embedding = row['embedding']
        library_unique[embedding] = uuid

        uuid_options = smi2options[smi]
        library_options[embedding] = uuid_options

    torch.save(library_unique, outdir / 'library_unique_emb2uuid.pt')
    torch.save(library_options, outdir / 'library_options_emb2uuid.pt')

    print(f"Full library: {len(library)} entries")
    print(f"Unique library: {len(library_unique)} entries")