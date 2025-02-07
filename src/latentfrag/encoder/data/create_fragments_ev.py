import argparse
import pandas as pd
import os
import torch

from rdkit import Chem
from tqdm import tqdm
from Bio.PDB import PDBParser

from latentfrag.encoder.data.fragmentation_utils import get_BRICS_fragments, stitch_small_fragments, clean_arom_fragment

protein_atom_mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "Se": 5}


p = argparse.ArgumentParser()
p.add_argument('--ligands', action='store', type=str,  required=True)
p.add_argument('--chains', action='store', type=str,  required=True)
p.add_argument('--out_dir', action='store', type=str,  required=True)
p.add_argument('--min_num_atoms', action='store', type=int,  required=False, default=1)
p.add_argument('--radius', action='store', type=float,  required=False, default=5)
p.add_argument('--limit', action='store', type=int,  required=False, default=None)


def get_close_chains(ligand_coords, pdb_name, pdb_dir, radius):
    # get all chains of the PDB in the dir
    chains = []
    ligand_coords = torch.from_numpy(ligand_coords).float()
    for fname in os.listdir(pdb_dir):
        if fname.startswith(pdb_name) and fname.endswith('.pdb'):
            chain_name = fname.split('.')[0].split('_')[-1]
            filepath = os.path.join(pdb_dir, fname)

            # load pdb and get coords
            parser = PDBParser(QUIET=True)
            model = parser.get_structure("", filepath)[0]
            coords = []
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element in protein_atom_mapping:
                            coords.append(torch.from_numpy(atom.get_coord()))

            coords = torch.stack(coords)
            # check if any atom is close to the ligand
            contact = torch.any(torch.cdist(ligand_coords, coords) < radius)
            if contact:
                chains.append(chain_name)
    return chains


def main(ligands_dir, chains_dir, out_dir, radius, limit):
    os.makedirs(out_dir, exist_ok=True)

    fragments = []
    fragments_ev = []
    processed_data = []

    uuid = 0
    for fnum, fname in enumerate(tqdm(os.listdir(ligands_dir)[:limit])):
        if not fname.endswith('.sdf'):
            continue

        pdb = fname.split('.')[0]
        if not len(pdb) == 4:
            continue

        supp = Chem.SDMolSupplier(os.path.join(ligands_dir, fname))

        for idx, ligand in enumerate(supp):
            bond_prop = ligand.GetProp('_Bonds')
            if bond_prop == 'RDKit':
                # skip ligands where bonds where not assigned from SMILES from PDB
                continue
            ligand_coords = ligand.GetConformer().GetPositions()

            # get chains that are close to the ligand
            chains = get_close_chains(ligand_coords, pdb, chains_dir, radius)

            frags, atom_mapping, frags_ev = get_BRICS_fragments(ligand)

            if args.min_num_atoms > 1:
                # check that ligand itself is not too small
                if ligand.GetNumHeavyAtoms() < args.min_num_atoms:
                    continue
                frag_indices = [i for i, frag in enumerate(frags) if frag.GetNumHeavyAtoms() < args.min_num_atoms]
                revised_frags = frags.copy()
                revised_atom_mapping = atom_mapping.copy()
                revised_frags_ev = frags_ev.copy()
                while frag_indices:
                    # check based on atom mapping if fragments were connected before
                    revised_frags_ev, revised_atom_mapping = stitch_small_fragments(revised_frags_ev, revised_atom_mapping, frag_indices, ligand)
                    # deduplicate revised_frags_ev
                    smiles_revised = [Chem.MolToSmiles(frag) for frag in revised_frags_ev]
                    smiles_revised_unique = list(set(smiles_revised))
                    unique_indices = [smiles_revised.index(smi) for smi in smiles_revised_unique]
                    revised_frags_ev = [revised_frags_ev[i] for i in unique_indices]
                    revised_atom_mapping = [revised_atom_mapping[i] for i in unique_indices]
                    revised_frags = [clean_arom_fragment(frag) for frag in revised_frags_ev]

                    frag_indices = [i for i, frag in enumerate(revised_frags) if frag.GetNumHeavyAtoms() < args.min_num_atoms]

            for frag, frag_ev in zip(revised_frags, revised_frags_ev):
                if frag is None:
                    continue

                for chain in chains:
                    frag.SetProp('_Chain', chain)
                    frag_ev.SetProp('_Chain', chain)
                    frag.SetProp('_PDB', pdb)
                    frag_ev.SetProp('_PDB', pdb)
                    frag.SetProp('_Idx', str(idx))
                    frag_ev.SetProp('_Idx', str(idx))
                    frag.SetProp('_UUID', str(uuid))
                    frag_ev.SetProp('_UUID', str(uuid))
                    fragments.append(frag)
                    fragments_ev.append(frag_ev)

                    processed_data.append({
                        'uuid': uuid,
                        'pdb': pdb,
                        'idx': idx,
                        'chain': chain,
                        'frag_smi': Chem.MolToSmiles(frag),
                        'frag_ev_smi': Chem.MolToSmiles(frag_ev),
                    })
                    uuid += 1

        if fnum % 5000 == 0:
            pd.DataFrame(processed_data).to_csv(f'{out_dir}/index.csv', index=False)
            with Chem.SDWriter(f'{out_dir}/fragments.sdf') as w:
                for frag in fragments:
                    w.write(frag)
            with Chem.SDWriter(f'{out_dir}/fragments_ev.sdf') as w:
                for frag in fragments_ev:
                    w.write(frag)

    pd.DataFrame(processed_data).to_csv(f'{out_dir}/index.csv', index=False)
    with Chem.SDWriter(f'{out_dir}/fragments.sdf') as w:
        for frag in fragments:
            w.write(frag)
    with Chem.SDWriter(f'{out_dir}/fragments_ev.sdf') as w:
        for frag in fragments_ev:
            w.write(frag)


if __name__ == '__main__':
    args = p.parse_args()
    main(
        ligands_dir=args.ligands,
        chains_dir=args.chains,
        out_dir=args.out_dir,
        radius=args.radius,
        limit=args.limit,
    )
