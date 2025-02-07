from pathlib import Path
from time import time
import argparse
import shutil
import random
import yaml
from collections import defaultdict

import torch
from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
import pandas as pd

from latentfrag.fm.data.data_utils import (process_raw_pair,
                                        get_n_nodes,
                                        get_type_histogram,
                                        rdmol_to_smiles,
                                        get_distribution,
                                        get_bit_histogram)
from latentfrag.fm.utils.gen_utils import write_sdf_file
from latentfrag.fm.utils.constants import atom_encoder
from latentfrag.encoder.models.lightning_modules import FragEmbed


def get_ligand_name(sdf_file, lig_idx):
    mol = Chem.SDMolSupplier(str(sdf_file))[lig_idx]
    ligand_name = mol.GetProp('_Name')
    return mol, ligand_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir', type=Path)
    parser.add_argument('struct_dir', type=Path)
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--pocket', type=str, default='CA+',
                        choices=['CA+', 'surface'])
    parser.add_argument('--ligand', type=str, default='latent', choices=['latent', 'fp'])
    parser.add_argument('--embedder_ckpt', type=Path, default=None)
    parser.add_argument('--min_frag_size', type=int, default=1)
    parser.add_argument('--cutoff', type=float, default=8.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--normal_modes', action='store_true')
    parser.add_argument('--msms_bin', type=Path, default=None)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--filter_by_nci', action='store_true')
    args = parser.parse_args()

    random.seed(args.random_seed)

    if args.ligand == 'latent':
        # Load fragment embedder model
        assert args.embedder_ckpt.exists(), f"Embedder checkpoint not found: {args.embedder_ckpt}"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedder_model = FragEmbed.load_from_checkpoint(args.embedder_ckpt, map_location=device, strict=False)
        embedder_model.to(device)
        embedder_model.eval()
        msms_resolution = embedder_model.hparams['surface_params']['resolution']

    if args.pocket == 'surface':
        assert args.msms_bin.exists(), f"MSMS binary not found: {args.msms_bin}"

    pdbdir = Path(args.struct_dir, 'chains_protonated')
    sdfdir = Path(args.struct_dir, 'ligands')

    # Make output directory
    dirname = f"processed_latentfbdd_{args.pocket}"
    if args.normal_modes:
        dirname += '_nma'
    if args.filter_by_nci:
        dirname += '_plipfil'
    if args.toy:
        dirname += '_toy'

    if args.ligand == 'fp':
        ckpt_name = 'fp'
    else:
        ckpt_name = Path(args.embedder_ckpt).parent.parent.name

    processed_dir = Path(args.csv_dir, dirname) if args.outdir is None else args.outdir
    processed_dir = processed_dir / ckpt_name
    processed_dir.mkdir(parents=True, exist_ok=args.resume)

    if args.pocket == 'surface':
        res_str = f"{msms_resolution:.1f}".replace('.', 'p')
        msms_processed_dir = Path(args.struct_dir, f'msms_{res_str}_chains_protonated')
        msms_processed_dir.mkdir(exist_ok=True)

    # Read data split
    data_split = {
        'train': pd.read_csv(Path(args.csv_dir, 'train_fbdd_ids.csv')),
        'val': pd.read_csv(Path(args.csv_dir, 'val_fbdd_ids.csv')),
        'test': pd.read_csv(Path(args.csv_dir, 'test_fbdd_ids.csv')),
    }

    # data_split['val'] = data_split['val'].sample(n=100, random_state=args.random_seed)
    # data_split['test'] = data_split['test'].sample(n=100, random_state=args.random_seed)
    # shuffle val and test set
    data_split['val'] = data_split['val'].sample(frac=1, random_state=args.random_seed)
    data_split['test'] = data_split['test'].sample(frac=1, random_state=args.random_seed)

    if args.toy:
        data_split['train'] = data_split['train'].sample(n=100, random_state=args.random_seed)
        data_split['val'] = data_split['val'].sample(n=10, random_state=args.random_seed)
        data_split['test'] = data_split['test'].sample(n=10, random_state=args.random_seed)


    failed = {}

    num_eval = 100

    for split in data_split.keys():
        if Path(processed_dir, f'train.pt').exists() and split == 'train' and args.resume:
            print(f"Skipping {split} dataset...")
            continue
        smiles = []
        frag_smiles = []

        print(f"Processing {split} dataset...")

        ligands = defaultdict(list)
        pockets = defaultdict(list)
        clusters = defaultdict(list)

        tic = time()
        _data = data_split[split]
        n = 0
        pbar = tqdm(zip(_data.pdb, _data.chain, _data.idx, _data.cluster), total=len(_data))
        for pdb_id, chain_id, lig_idx, cluster_id in pbar:
            if split != 'train' and n >= num_eval:
                break

            pbar.set_description(f'#failed: {len(failed)}')

            sdffile = sdfdir / f'{pdb_id}.sdf'
            pdbfile = pdbdir / f'{pdb_id}_{chain_id}.pdb'

            try:
                pdb_model = PDBParser(QUIET=True).get_structure('', pdbfile)[0]

                rdmol, ligand_name = get_ligand_name(sdffile, lig_idx)

                ligand_enum, pocket, frag_smi, fragments = process_raw_pair(
                    pdb_model,
                    rdmol,
                    frag_embedder=embedder_model if args.ligand == 'latent' else None,
                    dist_cutoff=args.cutoff,
                    pocket_representation=args.pocket,
                    pdb_file=pdbfile if args.normal_modes or args.pocket == 'surface' else None,
                    min_frag_size=args.min_frag_size,
                    msms_bin=str(args.msms_bin),
                    msms_resolution=msms_resolution if args.pocket == 'surface' else None,
                    crossdocked=False,
                    msms_processed_dir=msms_processed_dir if args.pocket == 'surface' else None,
                    filter_by_nci=args.filter_by_nci,)

                n += 1

            except (KeyError, AssertionError, FileNotFoundError, IndexError,
                    ValueError, AttributeError, RuntimeError, TimeoutError) as e:
                failed[(split, sdffile, pdbfile)] = (type(e).__name__, str(e))
                continue

            frag_smiles.extend(frag_smi)

            for ligand in ligand_enum:
                frag_keys = ['coarse_x', 'coarse_one_hot', 'coarse_bonds', 'coarse_bond_one_hot', 'axis_angle', 'frag_smiles', 'frag_smiles_ev',]
                surface_keys = ['normals_surface', 'x_surface', 'size_surface',
                                    'mask_surface', 'atom_xyz_surface', 'atomtypes_surface', 'atom_batch_surface']
                for k in ['x', 'one_hot', 'bonds', 'bond_one_hot', 'v', 'nma_vec'] + frag_keys + surface_keys:
                    if k in ligand:
                        ligands[k].append(ligand[k])
                    if k in pocket:
                        pockets[k].append(pocket[k])

                pocket_name = f"{pdb_id}_{chain_id}"
                ligands['name'].append(ligand_name)
                pockets['name'].append(pocket_name)
                smiles.append(rdmol_to_smiles(rdmol))

                # Add cluster info
                curr_idx = len(ligands['x']) - 1
                clusters[cluster_id].append(curr_idx)

                if split in {'val', 'test'}:
                    pdb_sdf_dir = processed_dir / split
                    pdb_sdf_dir.mkdir(exist_ok=True)

                    # Copy PDB file
                    pdb_file_out = Path(pdb_sdf_dir, pocket_name).with_suffix('.pdb')
                    shutil.copy(pdbfile, pdb_file_out)

                    # Copy SDF file
                    sdf_file_out = Path(pdb_sdf_dir, f"{pocket_name}_{ligand_name}").with_suffix('.sdf')
                    shutil.copy(sdffile, sdf_file_out)

                    # Write SDF file with fragments
                    frag_sdf_file = Path(pdb_sdf_dir, f"{pocket_name}_{ligand_name}_frags").with_suffix('.sdf')
                    write_sdf_file(frag_sdf_file, fragments)

        data = {'ligands': ligands, 'pockets': pockets, 'clusters': clusters}
        torch.save(data, Path(processed_dir, f'{split}.pt'))

        if split == 'train':
            np.save(Path(processed_dir, 'train_smiles.npy'), smiles)
            np.save(Path(processed_dir, 'train_smiles_frags.npy'), frag_smiles)
        elif split == 'val':
            np.save(Path(processed_dir, 'val_smiles.npy'), smiles)
            np.save(Path(processed_dir, 'val_smiles_frags.npy'), frag_smiles)
        elif split == 'test':
            np.save(Path(processed_dir, 'test_smiles.npy'), smiles)
            np.save(Path(processed_dir, 'test_smiles_frags.npy'), frag_smiles)

        print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")


    # --------------------------------------------------------------------------
    # Compute statistics & additional information
    # --------------------------------------------------------------------------
    train_data = torch.load(Path(processed_dir, f'train.pt'))

    # Maximum molecule size
    max_ligand_size = max([len(x) for x in train_data['ligands']['x']])

    # Joint histogram of number of ligand and pocket nodes
    pocket_coords = train_data['pockets']['x']
    ligand_coords = train_data['ligands']['x']
    frag_coords = train_data['ligands']['coarse_x']
    n_coarse_nodes = get_n_nodes(frag_coords, pocket_coords)
    np.save(Path(processed_dir, 'coarse_size_distribution.npy'), n_coarse_nodes)

    # Get histograms of ligand node types
    lig_one_hot = [x.numpy() for x in train_data['ligands']['one_hot']]
    ligand_hist = get_type_histogram(lig_one_hot, atom_encoder)
    np.save(Path(processed_dir, 'ligand_type_histogram.npy'), ligand_hist)

    # Coarse histograms
    coarse_embeddings = [x.detach().numpy() for x in train_data['ligands']['coarse_one_hot']]
    if args.ligand == 'latent':
        coarse_dist = get_distribution(coarse_embeddings)
        np.save(Path(processed_dir, 'coarse_emb_dist.npy'), coarse_dist)
    else:
        coarse_dist = get_bit_histogram(coarse_embeddings)
        np.save(Path(processed_dir, 'coarse_emb_dist.npy'), coarse_dist)

    # Write error report
    error_str = ""
    for k, v in failed.items():
        error_str += f"{'Split':<15}:  {k[0]}\n"
        error_str += f"{'Ligand':<15}:  {k[1]}\n"
        error_str += f"{'Pocket':<15}:  {k[2]}\n"
        error_str += f"{'Error type':<15}:  {v[0]}\n"
        error_str += f"{'Error msg':<15}:  {v[1]}\n\n"

    with open(Path(processed_dir, 'errors.txt'), 'w') as f:
        f.write(error_str)

    metadata = {
        'max_ligand_size': max_ligand_size
    }
    with open(Path(processed_dir, 'metadata.yml'), 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
