from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import FragmentOnBRICSBonds, GetMolFrags


def clean_arom_fragment(mol):
    # replace dummy atom with hydrogen instead of deleting it
    emol = Chem.EditableMol(mol)
    for i in range(mol.GetNumAtoms())[::-1]:
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 0:
            for bond in mol.GetAtomWithIdx(i).GetBonds():
                if bond.GetBondTypeAsDouble() != 1:
                    continue
                else:
                    emol.ReplaceAtom(i, Chem.Atom(1))
    return Chem.RemoveHs(emol.GetMol())


def get_src_atom_idx(a):
    '''https://github.com/UnixJunkie/molenc/blob/master/bin/molenc_smisur.py#L135'''
    neighbs = a.GetNeighbors()
    assert(len(neighbs) == 1)
    src_a = neighbs[0]
    return src_a.GetIdx()


def bind_molecules(m1, m2):
    '''https://github.com/UnixJunkie/molenc/blob/master/bin/molenc_smisur.py#L135'''
    n1 = m1.GetNumAtoms()
    n2 = m2.GetNumAtoms()
    m = n1 + n2
    rw_mol = Chem.RWMol(Chem.CombineMols(m1, m2))
    assert(rw_mol.GetNumAtoms() == m)
    indices = []
    for atom in rw_mol.GetAtoms():
        if atom.GetIsotope() == 7:
            indices.append(atom.GetIdx())
    assert len(indices) == 2
    ai = rw_mol.GetAtomWithIdx(indices[0])
    aj = rw_mol.GetAtomWithIdx(indices[1])
    src_idx = get_src_atom_idx(ai)
    src_idx2 = get_src_atom_idx(aj)
    bond_type_i = rw_mol.GetBondBetweenAtoms(src_idx, indices[0]).GetBondType()
    bond_type_j = rw_mol.GetBondBetweenAtoms(src_idx2, indices[1]).GetBondType()
    assert bond_type_i == bond_type_j != Chem.rdchem.BondType.SINGLE

    try:
        # attach. points are compatible
        rw_mol.AddBond(src_idx, src_idx2, bond_type_i)
        rw_mol.RemoveAtom(max(indices))
        rw_mol.RemoveAtom(min(indices))
        return rw_mol
    except:
        # attach. points are incompatible !!!
        print("bind_molecules: could not connect fragment")
        return None


def get_BRICS_fragments(mol):
    fragments = []
    corr_frags = []
    atom_mapping = []
    frags_ev = []
    corr_frags_ev = []

    for frag in GetMolFrags(FragmentOnBRICSBonds(mol), asMols=True, fragsMolAtomMapping=atom_mapping):
        # remove all indices bigger than atom num from mapping (dummies)
        num_atoms = mol.GetNumAtoms()
        atom_mapping_clean = []
        for indices in atom_mapping:
            frag_indices = []
            for idx in indices:
                if idx < num_atoms:
                    frag_indices.append(idx)
            atom_mapping_clean.append(frag_indices)

        frags_ev.append(frag)

        frag_clean = clean_arom_fragment(frag)
        try:
            Chem.SanitizeMol(frag_clean)
        except:
            continue
        fragments.append(frag_clean)

    # rest necessary as BRICS fragments double bonds, which is undesired here
    # also BRICS.BuildMolecules does not work on dummy 7
    have_dummy = {}
    atom_mapping_reordered = []
    frag_num = 0
    for frag, indices in zip(fragments, atom_mapping_clean):
        to_correct = False
        for a in frag.GetAtoms():
            if a.GetAtomicNum() == 0:
                isotope = a.GetIsotope()
                if isotope not in have_dummy:
                    have_dummy[isotope] = []
                have_dummy[isotope].append((frag, indices, frag_num))
                to_correct = True
                break
        if not to_correct:
            corr_frags.append(frag)
            corr_frags_ev.append(frags_ev[frag_num])
            atom_mapping_reordered.append(indices)
        frag_num += 1

    for isotope in have_dummy:
        if len(have_dummy[isotope]) == 2:
            frag1, idx1, frag_num1 = have_dummy[isotope][0]
            frag2, idx2, frag_num2 = have_dummy[isotope][1]
            frag = bind_molecules(frag1, frag2)
            frag_ev = bind_molecules(frags_ev[frag_num1], frags_ev[frag_num2])
            if frag is not None:
                corr_frags.append(frag)
                corr_frags_ev.append(frag_ev)
                atom_mapping_reordered.append(idx1 + idx2)

    return corr_frags, atom_mapping_reordered, corr_frags_ev


def get_identical_coords(mol):
    coords_rw = mol.GetConformer().GetPositions()
    identical_coords = []
    for i, coord in enumerate(coords_rw):
        for j, coord2 in enumerate(coords_rw):
            if i == j:
                continue
            if np.allclose(coord, coord2):
                ij_sorted = tuple(sorted([i, j]))
                if ij_sorted not in identical_coords:
                    identical_coords.append(ij_sorted)
    return identical_coords


def index_sorting(mol, identical_coords):
    idx_remove = []
    idx_connect = []
    for i in identical_coords[0]:
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 0:
            idx_remove.append(i)
        else:
            for j in identical_coords[1]:
                if mol.GetAtomWithIdx(j).GetAtomicNum() == 0:
                    idx_remove.append(j)
                else:
                    idx_connect.append((i, j))
    return idx_remove, idx_connect


def stitch_small_fragments(frags_ev, atom_mapping, frag_indices, mol):
    revised_frags_ev = [frag for i, frag in enumerate(frags_ev) if i not in frag_indices]
    revised_atom_mapping = [mapping for i, mapping in enumerate(atom_mapping) if i not in frag_indices]
    for i in frag_indices:
        frag_i = frags_ev[i]
        curr_mapping = atom_mapping[i]
        for j, frag_j in enumerate(frags_ev):
            if i == j:
                continue
            mapping2 = atom_mapping[j]
            # check if any bond between fragments before
            for idx1 in curr_mapping:
                for idx2 in mapping2:
                    if mol.GetBondBetweenAtoms(idx1, idx2) is not None:
                        # stitch back together
                        rw_mol = Chem.RWMol(Chem.CombineMols(frag_i, frag_j))

                        identical_coords = get_identical_coords(rw_mol)
                        if len(identical_coords) > 2:
                            continue

                        idx_remove, idx_connect = index_sorting(rw_mol, identical_coords)

                        rw_mol.AddBond(*idx_connect[0], Chem.rdchem.BondType.SINGLE)
                        rw_mol.RemoveAtom(max(idx_remove))
                        rw_mol.RemoveAtom(min(idx_remove))
                        revised_frags_ev.append(Chem.Mol(rw_mol))
                        # XXX the order in the fragments not matching anymore!
                        # can still use to infer coarse edges!
                        revised_atom_mapping.append(curr_mapping + mapping2)
    return revised_frags_ev, revised_atom_mapping

def get_stitched_frags_wo_ev(frags_ev, revised_atom_mapping):
    # smiles_revised = [Chem.MolToSmiles(frag) for frag in frags_ev]
    # smiles_revised_unique = list(set(smiles_revised))
    # unique_indices = [smiles_revised.index(smi) for smi in smiles_revised_unique]

    atom_mapping_sorted = [list(set(x)) for x in revised_atom_mapping]
    # deduplicate
    new_atom_mapping = []
    for elem in atom_mapping_sorted:
        if elem not in new_atom_mapping:
            new_atom_mapping.append(elem)
    unique_indices = [atom_mapping_sorted.index(x) for x in new_atom_mapping]

    frags_ev = [frags_ev[i] for i in unique_indices]
    revised_atom_mapping = [revised_atom_mapping[i] for i in unique_indices]
    revised_frags = [clean_arom_fragment(frag) for frag in frags_ev]

    return revised_frags, revised_atom_mapping, frags_ev


def combine_fragments(fragments):
    n = min(len(fragments), 10)
    all_indices = set().union(*fragments)

    def is_valid_combination(combo):
        covered_indices = set()
        for fragment in combo:
            if any(idx in covered_indices for idx in fragment):
                return False
            covered_indices.update(fragment)
        return covered_indices == all_indices

    valid_combinations = []

    for r in range(1, n + 1):
        for combo in combinations(fragments, r):
            if is_valid_combination(combo):
                valid_combinations.append(list(combo))

    return valid_combinations
