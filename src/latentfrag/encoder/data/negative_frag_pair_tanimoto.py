from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=Path, required=True)

args = parser.parse_args()

assert args.datadir.exists()

CUTOFF = 0.1

# Load data
df_train = pd.read_csv(args.datadir / 'index_train.csv')
df_test = pd.read_csv(args.datadir / 'index_test.csv')
df_val = pd.read_csv(args.datadir / 'index_valid.csv')

rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath = 2, fpSize = 2048)

splits = ['train', 'test', 'valid']

for df, split in zip([df_train, df_test, df_val], splits):
    print(f'Processing {split} set')

    df_dedup = df.drop_duplicates(subset='frag_smi')

    smiles = df_dedup['frag_smi'].tolist()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [rdkgen.GetFingerprint(mol) for mol in mols]

    # get indices below cut-off for every molecule
    indices = []
    for i in range(len(fps)):
        similarity = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        indices.append(np.where(np.array(similarity) < CUTOFF)[0])

    uuids = df_dedup['uuid'].tolist()
    dissimilar_uuids = [np.array(uuids)[i] for i in indices]

    smi_to_dissimilar_uuids = dict(zip(smiles, dissimilar_uuids))
    df['dissimilar_uuids'] = df['frag_smi'].map(smi_to_dissimilar_uuids)
    uuid_to_dissimilar_uuids = dict(zip(df['uuid'], df['dissimilar_uuids']))

    # save dict
    np.save(args.datadir / f'dissimilar_uuids_{split}.npy', uuid_to_dissimilar_uuids)

print('Done')
