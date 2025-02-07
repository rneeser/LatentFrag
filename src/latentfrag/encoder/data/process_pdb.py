import argparse
import os
import io
import numpy as np
import pandas as pd
import prody

from rdkit import rdBase, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit import Chem

from latentfrag.encoder.data.const import ALLOWED_PROTEIN_ATOMS, ALLOWED_PROTEIN_RESIDUES, MOAD_INVALID_COMPOUNDS


def is_valid_chain(chain):
    for atom in chain:
        if atom.getElement() not in ALLOWED_PROTEIN_ATOMS:
            return False
        if atom.getResname() not in ALLOWED_PROTEIN_RESIDUES:
            return False
    return True


def is_valid_ligand(ligand):
    if ligand is None:
        # print('was terminal')
        return False

    mass = sum(ligand.getMasses())
    if mass < 100:
        return False

    resnames = list(set(ligand.getResnames()))
    if len(resnames) != 1:
        return False

    if resnames[0] in ALLOWED_PROTEIN_RESIDUES:
        return False

    if resnames[0] in MOAD_INVALID_COMPOUNDS:
        return False

    return True


def create_ligand_name(ligand):
    chid = ligand.getChids()[0]
    resnum = ligand.getResnums()[0]
    cci = ligand.getResnames()[0]
    return f'{cci}_{resnum}_{chid}'


def assign_bonds(ligand, cci2smi, pdb_code):
    name = create_ligand_name(ligand)
    cci = ligand.getResnames()[0]
    smi = cci2smi.get(cci)

    output = io.StringIO()
    prody.writePDBStream(output, ligand)
    rd_ligand = AllChem.MolFromPDBBlock(output.getvalue(), sanitize=False)
    rd_ligand = Chem.RemoveAllHs(rd_ligand, sanitize=False)
    if rd_ligand is None:
        print(f'{pdb_code}: Could not create RDMol from PDB block')
        return None

    assigned_pdb_bonds = False
    if smi is not None:
        try:
            template = AllChem.MolFromSmiles(smi, sanitize=False)
            t = Chem.RemoveAllHs(template, sanitize=False)
            rd_ligand = AllChem.AssignBondOrdersFromTemplate(t, rd_ligand)
            assigned_pdb_bonds = True
        except ValueError as e:
            # print(f'Could not find matching for ligand {pdb_code}_{name}: {e}')
            pass
        except AssertionError:
            # print(f'Template for ligand {pdb_code}_{name} is None')
            pass
        except Exception as e:
            print(f'{pdb_code}: Unexpected error for ligand {name}: {e}')
            pass
    else:
        # print(f'Could not find ligand {pdb_code}_{name} in Ligand Expo SMILES dictionary)')
        pass

    rd_ligand.SetProp('_Name', name)
    rd_ligand.SetProp('_Bonds', 'PDB' if assigned_pdb_bonds else 'RDKit')

    # if not assigned_pdb_bonds:
    #     return None

    try:
        AllChem.SanitizeMol(rd_ligand)
    except:
        # print('not sanitized molecule - not saved! ')
        return None

    if rd_ligand.GetNumAtoms() > 70:
        return None

    return rd_ligand


def clean_and_split_protein(in_pdb_path, cci2smi, pdb_code=None):
    data = prody.parsePDB(in_pdb_path)
    csid = data.getACSIndex()

    protein = data.select('not water and protein and heavy')
    ligands = data.select('hetatm and not water and not ion and heavy')

    if protein is None:
        # print(f'{pdb_code}: protein is None')
        try:
            data = prody.parsePDB(in_pdb_path, model=2)
            csid = data.getACSIndex()
            protein = data.select('not water and protein and heavy')
            ligands = data.select('hetatm and not water and not ion and heavy')
        except:
            return dict(), dict(), csid
    if ligands is None:
        # print(f'{pdb_code}: ligands is None')
        return dict(), dict(), csid

    valid_ligands = dict()
    for chid, resnum in set(zip(ligands.getChids(), ligands.getResnums())):
        try:
            ligand = ligands.select(f'chain {chid} and resnum {resnum}')
        except:
            ligand = None
            print('Ligand - is probably terminal aminoacid modification, skipped')

        if is_valid_ligand(ligand):
            name = create_ligand_name(ligand)
            rd_ligand = assign_bonds(ligand, cci2smi=cci2smi, pdb_code=pdb_code)
            if rd_ligand is not None:
                valid_ligands[name] = rd_ligand

    if len(valid_ligands) == 0:
        # print(f'{pdb_code}: no valid ligands')
        return dict(), dict(), csid

    valid_chains = dict()
    for chid in set(protein.getChids()):
        chain = protein.select(f'chain {chid}')
        if is_valid_chain(chain):
            valid_chains[chid] = chain

    return valid_ligands, valid_chains, csid


def compute_interactions(ligands, chains):
    chains_to_save = set()
    ligands_to_save = set()
    interactions = []
    for chid, chain in chains.items():
        chain_coords = chain.getCoords()
        for ligand_name, ligand in ligands.items():
            ligand_coords = ligand.GetConformer().GetPositions()
            distances = np.linalg.norm(ligand_coords[None, :, :] - chain_coords[:, None, :], axis=-1)
            contacts = (distances <= 5.).any(axis=0).sum()  # Number of ligand atoms in contact
            num_atoms = ligand.GetNumAtoms()
            assert contacts <= num_atoms

            if contacts > 0:
                chains_to_save.add(chid)
                ligands_to_save.add(ligand_name)
                interactions.append({
                    'chain': chid,
                    'ligand': ligand_name,
                    'bonds_src': ligand.GetProp('_Bonds'),
                    'n_contact_atoms': contacts,
                    'n_atoms': num_atoms,
                })

    return ligands_to_save, chains_to_save, interactions


def process(pdb_dir, ligand_expo_smiles, out_chains_dir, out_ligands_dir, out_interactions_table, enable_resume):
    os.makedirs(out_chains_dir, exist_ok=True)
    os.makedirs(out_ligands_dir, exist_ok=True)

    smiles_table = pd.read_csv(ligand_expo_smiles, sep='\t', names=['smi', 'cci', 'name'], )
    smiles_table = smiles_table[~smiles_table.smi.isna()]
    cci2smi = dict(smiles_table[['cci', 'smi']].values)

    processed_pdb_codes = set()
    interactions_table = None
    if enable_resume and os.path.exists(out_interactions_table):
        interactions_table = pd.read_csv(out_interactions_table)
        processed_pdb_codes = set(interactions_table.pdb.unique().tolist())
        print(f'Found {len(processed_pdb_codes)} files processed')

    all_interactions = []
    for fname in tqdm(os.listdir(pdb_dir)):
        pdb_code = fname.split('.')[0]
        if not fname.endswith('.pdb.gz') or len(pdb_code) != 4 or pdb_code in processed_pdb_codes:
            continue

        in_pdb_path = os.path.join(pdb_dir, fname)
        try:
            valid_ligands, valid_chains, csid = clean_and_split_protein(
                in_pdb_path=in_pdb_path,
                cci2smi=cci2smi,
                pdb_code=pdb_code
            )
        except Exception as e:
            print(f'{pdb_code}: unexpected error when splitting the file: {e}')
            continue

        ligands_to_save, chains_to_save, interactions = compute_interactions(valid_ligands, valid_chains)
        if len(ligands_to_save) == 0 or len(chains_to_save) == 0 or len(interactions) == 0:
            assert len(ligands_to_save) == len(chains_to_save) == len(interactions)
            continue

        for interaction in interactions:
            all_interactions.append({'pdb': pdb_code, **interaction})

        all_interactions_table = pd.DataFrame(all_interactions)
        if enable_resume and os.path.exists(out_interactions_table):
            all_interactions_table = pd.concat([interactions_table, all_interactions_table])
        all_interactions_table.to_csv(out_interactions_table, index=False)

        out_ligands_path = os.path.join(out_ligands_dir, f'{pdb_code}.sdf')
        with AllChem.SDWriter(out_ligands_path) as writer:
            for ligand_name in ligands_to_save:
                writer.write(valid_ligands[ligand_name])

        for chid in chains_to_save:
            out_chain_path = os.path.join(out_chains_dir, f'{pdb_code}_{chid}.pdb')
            prody.writePDB(out_chain_path, valid_chains[chid], csets=csid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', action='store', type=str, required=True)
    parser.add_argument('--ligand_expo_smiles', action='store', type=str, required=True)
    parser.add_argument('--out_chains_dir', action='store', type=str, required=True)
    parser.add_argument('--out_ligands_dir', action='store', type=str, required=True)
    parser.add_argument('--out_interactions_table', action='store', type=str, required=True)
    parser.add_argument('--enable_resume', action='store_true', default=False)
    args = parser.parse_args()

    # Disable ProDy and RDKit logs
    prody.confProDy(verbosity='none')
    RDLogger.logger().setLevel(RDLogger.ERROR)
    rdBase.DisableLog('rdApp.error')

    process(
        pdb_dir=args.pdb_dir,
        ligand_expo_smiles=args.ligand_expo_smiles,
        out_chains_dir=args.out_chains_dir,
        out_ligands_dir=args.out_ligands_dir,
        out_interactions_table=args.out_interactions_table,
        enable_resume=args.enable_resume,
    )
