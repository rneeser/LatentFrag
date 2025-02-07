from itertools import product, combinations
from functools import reduce

import torch
import torch.nn.functional as F
from rdkit.Geometry import Point3D
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

from latentfrag.fm.utils.constants import (FLOAT_TYPE,
                                        INT_TYPE,
                                        atom_encoder,
                                        bond_encoder,
                                        frag_conn_decoder,)
from latentfrag.fm.data.data_utils import encode_atom
from latentfrag.fm.data.fragment import clean_arom_fragment
from latentfrag.fm.data.frag_frames import set_frame, generate_local_frame_aa
import latentfrag.fm.utils.so3_utils as so3


def set_coords(mol, new_coords):
    conf = mol.GetConformer()
    for i, coord in enumerate(new_coords):
        conf.SetAtomPosition(i, (coord[0].item(),
                                 coord[1].item(),
                                 coord[2].item()))
    return mol


def get_coords_rdmol(mol):
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    return coords


def com_from_rdmol(mol, use_ev=True):
    if not use_ev:
        mol = clean_arom_fragment(mol)
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    com = coords.mean(0)
    return com


def latent2frags(sampled_graph, library, fragments, needed_degree, k=1, rotate=False):
    '''
    Query the library for the fragment graph that is closest (cosine similarity)
    to the given embedding.
    '''
    embeddings = sampled_graph[1]
    com = sampled_graph[0]
    rot_vecs = sampled_graph[-1]

    # Get the embeddings of all fragments in the library
    lib_embeddings = list(library.keys())
    lib_embeddings = torch.stack(lib_embeddings) # m x 1 x 16

    similarities = F.cosine_similarity(embeddings.unsqueeze(0), lib_embeddings, dim=2)

    # Get the indices of the fragments with the top k highest similarity
    sims, indices = torch.topk(similarities, k=k, dim=0)
    # TODO filter out by similarity threshold?

    # Get the fragment uuids
    lib_uuids = list(library.values())

    # get all possiblities with different exit vector options
    frags_all = []
    for sub_idx in indices:
        sub_uuids = []
        for idx in sub_idx:
            sub_uuids.append(lib_uuids[idx])
        frags_all.append([fragments[uuid] for uuid in sub_uuids])

    # Get the fragments
    # choose the best combination based on degrees
    frags = np.zeros(indices.shape[1], dtype=object)
    for frag in frags_all:
        for i in range(len(frag)):
            if frags[i] == 0:
                frag_i = frag[i]
                deg = needed_degree.get(i, 0)
                if check_degree_one_frag(frag_i, deg):
                    frags[i] = frag_i
    # fill up positions that are still zero with highest similarity fragment
    for i in range(len(frags)):
        if frags[i] == 0:
            frags[i] = frags_all[0][i]

    # Translation and rotation (if needed) based on predicted modalities
    if rotate:
        local_frames = np.array([generate_local_frame_aa(frag) for frag in frags])
        local_frames = torch.tensor(local_frames, dtype=FLOAT_TYPE)
        old_rot_vecs = so3.rotation_vector_from_matrix(local_frames)
    else:
        old_rot_vecs = None
    old_coms = np.array([com_from_rdmol(frag) for frag in frags])
    old_coords = [get_coords_rdmol(frag) for frag in frags]
    old_coms = torch.tensor(old_coms, dtype=FLOAT_TYPE)
    # padding done to process all at once
    max_coords = max([len(coord) for coord in old_coords])
    old_coords_padded = [np.pad(coord, ((0, max_coords - len(coord)), (0, 0))) for coord in old_coords]
    old_coords_padded = torch.tensor(np.array(old_coords_padded), dtype=FLOAT_TYPE)
    new_coords = set_frame(old_coords_padded, old_coms, com, old_rot_vecs, rot_vecs, rotate=rotate)
    new_coords = [coord[:len(old_coord)] for coord, old_coord in zip(new_coords, old_coords)]

    for frag, new_coord in zip(frags, new_coords):
        frag = set_coords(frag, new_coord)

    return [frags]


def latent2fragcombos(sampled_graph, library, fragments, rotate=False):
    '''
    Query the library for the fragment graph that is closest (cosine similarity)
    to the given embedding.
    '''
    embeddings = sampled_graph[1]
    com = sampled_graph[0]
    rot_vecs = sampled_graph[-1]

    # Get the embeddings of all fragments in the library
    lib_embeddings = list(library.keys())
    lib_embeddings = torch.stack(lib_embeddings) # m x 1 x 16

    similarities = F.cosine_similarity(embeddings.unsqueeze(0), lib_embeddings, dim=2)

    # Get the indices of the fragments with the highest similarity
    indices = torch.argmax(similarities, dim=0)

    # Get the fragment uuids
    lib_uuids = list(library.values())

    # get all possiblities with different exit vector options
    uuids = [set(lib_uuids[idx]) for idx in indices]

    # Get the fragments
    uuid_combos = []
    for combination in combinations(uuids, r=embeddings.shape[0]):
        uuid_combos.extend(p for p in product(*combination))

    frag_combos = []
    for uuid_combo in uuid_combos:
        frags = [fragments[uuid] for uuid in uuid_combo]

        # Translation and rotation (if needed) based on predicted modalities
        if rotate:
            local_frames = np.array([generate_local_frame_aa(frag, use_ev=False) for frag in frags])
            local_frames = torch.tensor(local_frames, dtype=FLOAT_TYPE)
            old_rot_vecs = so3.rotation_vector_from_matrix(local_frames)
        else:
            old_rot_vecs = None
        old_coms = np.array([com_from_rdmol(frag, use_ev=False) for frag in frags])
        old_coords = [get_coords_rdmol(frag) for frag in frags]
        old_coms = torch.tensor(old_coms, dtype=FLOAT_TYPE)
        # padding done to process all at once
        max_coords = max([len(coord) for coord in old_coords])
        old_coords_padded = np.array([np.pad(coord, ((0, max_coords - len(coord)), (0, 0))) for coord in old_coords])
        old_coords_padded = torch.tensor(old_coords_padded, dtype=FLOAT_TYPE)
        new_coords = set_frame(old_coords_padded, old_coms, com, old_rot_vecs, rot_vecs, rotate=rotate)
        new_coords = [coord[:len(old_coord)] for coord, old_coord in zip(new_coords, old_coords)]

        frags_moved = []
        for frag, new_coord in zip(frags, new_coords):
            frag = set_coords(frag, new_coord)
            frags_moved.append(frag)

        frag_combos.append(frags_moved)

    return frag_combos #, graphs_moved


def get_graph_from_frag_ev(mol, indices):

    ligand_coord = mol.GetConformer().GetPositions()
    ligand_coord = torch.from_numpy(ligand_coord[indices]).to(dtype=FLOAT_TYPE)

    # Features
    ligand_onehot = F.one_hot(
        torch.tensor([encode_atom(a, atom_encoder) for a in mol.GetAtoms() if a.GetSymbol() not in ['R', '*']]),
        num_classes=len(atom_encoder)
    )

    num_atoms = len(ligand_coord)

    adj = np.ones((num_atoms, num_atoms)) * bond_encoder['NOBOND']
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if i in indices and j in indices:
            adj[i, j] = bond_encoder[str(b.GetBondType())]
            adj[j, i] = adj[i, j]  # undirected graph

    # molecular graph is undirected -> don't save redundant information
    bonds = np.stack(np.triu_indices(len(ligand_coord), k=1), axis=0)
    bond_types = adj[bonds[0], bonds[1]].astype('int64')
    bonds = torch.from_numpy(bonds)
    bond_types = F.one_hot(torch.from_numpy(bond_types), num_classes=len(bond_encoder))

    graph = {
        'x': ligand_coord.to(dtype=FLOAT_TYPE),
        'one_hot': ligand_onehot.to(dtype=FLOAT_TYPE),
        'mask': torch.zeros(len(ligand_coord), dtype=INT_TYPE),
        'bonds': bonds.to(INT_TYPE),
        'bond_one_hot': bond_types.to(FLOAT_TYPE),
        'bond_mask': torch.zeros(bonds.size(1), dtype=INT_TYPE),
        'size': torch.tensor([len(ligand_coord)], dtype=INT_TYPE),
        'n_bonds': torch.tensor([len(bond_types)], dtype=INT_TYPE),
    }
    return graph


def get_predicted_degrees(edges, edge_types):
    predicted_degrees = {}
    single_bonds = []
    for bond, bond_type in zip(edges.T, edge_types):
        bond_type = frag_conn_decoder[bond_type]
        src = bond[0].item()
        dst = bond[1].item()
        if str(bond_type) == 'SINGLE':
            predicted_degrees[src] = predicted_degrees.get(src, 0) + 1
            predicted_degrees[dst] = predicted_degrees.get(dst, 0) + 1
            single_bonds.append((src, dst, bond_type))
    return predicted_degrees, single_bonds


def check_degrees(frags_ev, needed_degree, single_bonds):
    if len(single_bonds) == 0:
        return False
    for i in range(len(frags_ev)):
        frag_i = frags_ev[i]
        deg = needed_degree.get(i, 0)
        if not check_degree_one_frag(frag_i, deg):
            return False
    # check if all nodes are connected (not isolated)
    num_nodes = len(frags_ev)
    num_bonds = len(single_bonds)
    if num_bonds < num_nodes - 1:
        return False
    return True


def check_degree_one_frag(frag_ev, degree):
    num_ev = 0
    for a in frag_ev.GetAtoms():
        if a.GetAtomicNum() == 0:
            num_ev += 1
    if num_ev != degree:
        return False
    return True


def combine_frags(frags_ev, needed_degree, single_bonds):
    mol_grow = None
    frags_in = set()
    num_atoms = 0

    passing_check = check_degrees(frags_ev, needed_degree, single_bonds)
    if not passing_check:
        return None

    for src, dst, bond_type in single_bonds:
        if src in frags_in and dst in frags_in:
            continue

        if bond_type is None:
            continue

        last_iter = False
        if str(bond_type) == 'SINGLE':
            if src in frags_in:
                frag_dst = frags_ev[dst]
                frag_src = mol_grow
                frags_in.add(dst)
            elif dst in frags_in:
                frag_dst = frags_ev[src]
                frag_src = mol_grow
                frags_in.add(src)
            elif dst not in frags_in and src not in frags_in and mol_grow is None:
                frag_src = [frags_ev[src]]
                frag_dst = frags_ev[dst]
                frags_in.add(src)
                frags_in.add(dst)
                num_atoms += frag_src[0].GetNumHeavyAtoms()
            else:
                # would start growing new sup-fragment, attach to queue
                single_bonds.append((src, dst, bond_type))
                continue

            if len(frags_in) == len(frags_ev):
                last_iter = True
            num_atoms += frag_dst.GetNumHeavyAtoms()
            mol_grow_raw = list(BRICS.BRICSBuild(frag_src + [frag_dst], onlyCompleteMols=last_iter, scrambleReagents=False, maxDepth=0))

            if len(mol_grow_raw) == 0:
                return None

            # check if number of atoms checks out
            mol_grow_raw = [m for m in mol_grow_raw if m.GetNumHeavyAtoms() == num_atoms]

            if len(mol_grow_raw) == 0:
                return None

            # check if all frags are in the mol_grow
            p = Chem.AdjustQueryParameters.NoAdjustments()
            p.makeDummiesQueries = True
            mol_grow = []
            frags_ev_query = []
            for frag in [frags_ev[dst], frags_ev[src]]:
                frag_copy = Chem.Mol(frag)
                for a in frag_copy.GetAtoms():
                    if a.GetAtomicNum() == 0:
                        a.SetIsotope(0)
                hq = Chem.AdjustQueryProperties(frag_copy, p)
                frags_ev_query.append(hq)
            for m in mol_grow_raw:
                frag_matches = []
                for frag in frags_ev_query:
                    frag_matches.append(m.HasSubstructMatch(frag, useChirality=True))
                if all(frag_matches):
                    mol_grow.append(m)

            if len(mol_grow) == 0:
                return None

    candidate_mols = list(mol_grow)

    if len(candidate_mols) == 0:
        return None

    # deduplicate mols based on SMILES
    smiles = [Chem.MolToSmiles(m) for m in candidate_mols]
    unique_smiles = list(set(smiles))
    unique_indices = [smiles.index(smile) for smile in unique_smiles]
    candidate_mols = [candidate_mols[idx] for idx in unique_indices]

    for mol in candidate_mols:
        assert all([atom.GetSymbol() != 'R' for atom in mol.GetAtoms()]), f'Molecule {Chem.MolToSmiles(mol)} has dummy atom; frags: {[Chem.MolToSmiles(frag) for frag in frags_ev]}'

    return candidate_mols


def get_candidate_mols(frag_combos, needed_degree, single_bonds):
    all_candidates = []
    all_frag_smiles = []
    for combo in frag_combos:
        candidate_mols = combine_frags(combo, needed_degree, single_bonds)
        if candidate_mols is not None:
            all_candidates.extend(candidate_mols)
            clean_frags = [clean_arom_fragment(frag) for frag in combo]
            all_frag_smiles.append([Chem.MolToSmiles(frag) for frag in clean_frags])

    # if no candidates found, return the original frags without exit vectors (for stats)
    if not all_candidates:
        combo = [clean_arom_fragment(frag) for frag in frag_combos[0]]
        combined_mol = reduce(Chem.CombineMols, combo)
        all_candidates.append(combined_mol)
        all_frag_smiles.append([Chem.MolToSmiles(frag) for frag in combo])
    return all_candidates, all_frag_smiles
