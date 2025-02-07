from pathlib import Path
import argparse

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
from tqdm import tqdm

from latentfrag.encoder.utils.data import TensorDict, featurize_ligand
from latentfrag.encoder.models.lightning_modules import FragEmbed
from latentfrag.fm.utils.constants import FLOAT_TYPE


parser = argparse.ArgumentParser()

parser.add_argument('--datadir', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--data_source', type=str, default='index_filtered_all.csv')

args = parser.parse_args()


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FragEmbed.load_from_checkpoint(args.ckpt, map_location=device)
model.to(device)
model.eval()

use_ev = model.hparams.get('ligand_encoder_params').get('use_ev', False)

# define paths
csv_path = args.datadir / args.data_source

sdf_path = args.datadir / 'fragments.sdf'
frags = Chem.SDMolSupplier(str(sdf_path), sanitize=True)
if use_ev:
    sdf_path_ev = args.datadir / 'fragments_ev.sdf'
    frags_ev = Chem.SDMolSupplier(str(sdf_path_ev), sanitize=True)
else:
    frags_ev = frags

ckpt_name = Path(args.ckpt).parent.parent.name
outdir = args.out / ckpt_name
outdir.mkdir(exist_ok=True, parents=True)

# load data
df = pd.read_csv(csv_path)

# get relevant data for library
df = df[['uuid', 'pdb', 'idx', 'chain', 'frag_smi', 'frag_ev_smi']]

library = {}

for uuid, pdbid, idx, chain, frag_smi, frag_ev_smi in tqdm(df.values):
    # graph, embedding = get_graph_embedding(mol, model)
    embedding = get_embedding(frags_ev[uuid], model)

    mol = frags[uuid]
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
    energy = ff.CalcEnergy()

    library[uuid] = {
        'pdbid': pdbid,
        'idx': idx,
        'chain': chain,
        'smiles': frag_smi,
        'smiles_ev': frag_ev_smi,
        # 'graph': graph,
        'embedding': embedding.to(FLOAT_TYPE),
        'energy': energy,
    }

# save full library
torch.save(library, outdir / 'library_all_uuid2data.pt')

library_df = pd.DataFrame(library).T
library_df['uuid'] = library_df.index
library_df = library_df.astype({'energy': float})
if not use_ev:
    library_df_sorted = library_df.sort_values(by=['energy','smiles','smiles_ev'], ascending=[True, True, True])
    library_df_dedup = library_df_sorted.drop_duplicates(subset=['smiles', 'smiles_ev'], keep='first')
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

else:
    library_df_sorted = library_df.sort_values(by=['energy','smiles_ev'], ascending=[True, True])
    library_df_dedup = library_df_sorted.drop_duplicates(subset=['smiles_ev'], keep='first')
    # drop rows with no dummy atom in smiles_ev
    library_df_dedup = library_df_dedup[library_df_dedup['smiles_ev'].str.contains(r'\*')]
    library_df_unique = library_df_dedup.groupby('smiles_ev').first()

    library_unique = {}
    for smi, row in library_df_unique.iterrows():
        uuid = row['uuid']
        embedding = row['embedding']
        library_unique[embedding] = uuid

    torch.save(library_unique, outdir / 'library_ev_unique_emb2uuid.pt')


# unique_uuids = list(library_unique.values())
# frag_library_graphs = {uuid: frag_library_graphs[uuid]['graph'] for uuid in unique_uuids}

# torch.save(frag_library_graphs, outdir / 'library_unique_uuid2graph.pt')

print(f"Full library: {len(library)} entries")
print(f"Unique library: {len(library_unique)} entries")
