'''
Code adapted from:
DrugFlow by A. Schneuing & I. Igashov
https://github.com/LPDI-EPFL/DrugFlow
'''
from itertools import accumulate, chain
import signal
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from Bio.PDB import StructureBuilder, PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from scipy.ndimage import gaussian_filter

from latentfrag.fm.data.normal_modes import pdb_to_normal_modes
from latentfrag.fm.data.fragment import (get_BRICS_fragments,
                                      stitch_small_fragments,
                                      get_stitched_frags_wo_ev,
                                      combine_fragments)
from latentfrag.fm.data.msms import pdb_to_points_normals, default_filter
from latentfrag.fm.data.frag_frames import generate_local_frame_aa
from latentfrag.fm.utils.constants import (FLOAT_TYPE,
                                  INT_TYPE,
                                  atom_encoder,
                                  bond_encoder,
                                  aa_atom_index,
                                  aa_encoder,
                                  residue_bond_encoder,
                                  frag_conn_encoder,
                                  protein_atom_mapping,)
from latentfrag.fm.utils.gen_utils import batch_to_list, batch_to_list_for_indices
import latentfrag.fm.utils.so3_utils as so3
from latentfrag.fm.analysis.interactions_plip import INTERACTION_LIST, prepare_mol, combine_mol_pdb, run_plip, read_plip
from latentfrag.encoder.utils.data import featurize_ligand


class TensorDict(dict):
    def __init__(self, **kwargs):
        super(TensorDict, self).__init__(**kwargs)

    def to(self, device):
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = v.to(device)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')


def encode_atom(rd_atom, atom_encoder):
    element = rd_atom.GetSymbol().capitalize()

    explicitHs = rd_atom.GetNumExplicitHs()
    if explicitHs == 1 and f'{element}H' in atom_encoder:
        return atom_encoder[f'{element}H']

    charge = rd_atom.GetFormalCharge()
    if charge == 1 and f'{element}+' in atom_encoder:
        return atom_encoder[f'{element}+']
    if charge == -1 and f'{element}-' in atom_encoder:
        return atom_encoder[f'{element}-']

    return atom_encoder[element]


def encode_fragment(frag, frag_ev, model):

    if model is not None:
        device = model.device
        use_ev = model.hparams.get('ligand_encoder_params').get('use_ev', False)

        frag_feat = featurize_ligand(frag_ev if use_ev else frag,
                                    use_ev=use_ev)
        frag_feat = TensorDict(**frag_feat)
        frag_feat['batch'] = torch.zeros(len(frag_feat['xyz']), dtype=int)
        frag_feat = frag_feat.to(device)

        _, frag_feat['frag_desc'] = model.ligand_encoder(
                frag_feat["xyz"], frag_feat["types"], frag_feat["batch"],
                frag_feat["bonds"], frag_feat["bond_types"], frag_feat["mol_feats"], return_global=True)

        frag_feat['geom_mean'] = torch.mean(frag_feat['xyz'], dim=0, dtype=FLOAT_TYPE)

        return frag_feat.cpu()

    else:
        coords = torch.from_numpy(frag_ev.GetConformer().GetPositions()).float()
        geom_mean = torch.mean(coords, dim=0, dtype=FLOAT_TYPE)
        feats = get_fingerprint(frag_ev, use_ev=True)

        return {
            'frag_desc': torch.from_numpy(feats).float().unsqueeze(0),
            'geom_mean': geom_mean,
        }


def custom_atom_invariants(mol):
    invariants = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # It's a dummy atom
            # Extract the digit from the isotope information
            digit = atom.GetIsotope()
            # Assign a unique invariant based on the digit
            # We add 200 to ensure it's distinct from other elements
            invariants.append(200 + digit)
        else:
            # For non-dummy atoms, use the atomic number as invariant
            invariants.append(atom.GetAtomicNum())
    return invariants


def get_fingerprint(mol, radius=3, nBits=64, use_ev=False):
    if use_ev:
        # Generate Morgan fingerprint with custom atom invariants
        invariants = custom_atom_invariants(mol)
    else:
        invariants = None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, invariants=invariants)

    # convert to numpy array
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_num_nci(frag, protein):
    if protein is None:
        return 0
    profile = {i: 0 for i in INTERACTION_LIST}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_pdb = prepare_mol(frag, Path(tmpdir))
            complex_pdb = Path(tmpdir, 'complex.pdb')
            combine_mol_pdb(mol_pdb, protein, str(complex_pdb))
            plip_out = run_plip(complex_pdb, Path(tmpdir), plip_exec='plipcmd')
            profile = read_plip(plip_out / 'report.xml')
            num_interactions = sum([profile[i] for i in INTERACTION_LIST])
    except Exception as e:
        print(f'Error in PLIP: {e}')
        num_interactions = 0
    return num_interactions


def prepare_ligand(rdmol,
                   atom_encoder,
                   bond_encoder,
                   frag_embedder,
                   min_frag_size=1,
                   filter_by_nci=False,
                   prot=None):

    # Coordinates
    ligand_coord = rdmol.GetConformer().GetPositions()
    ligand_coord = torch.from_numpy(ligand_coord)

    # Features
    ligand_onehot = F.one_hot(
        torch.tensor([encode_atom(a, atom_encoder) for a in rdmol.GetAtoms()]),
        num_classes=len(atom_encoder)
    )

    # fragments
    combos, fragments, frags_ev, atom_mapping = fragment_and_augment(rdmol, min_frag_size)

    ligands = []
    smiles = []
    if filter_by_nci:
        num_nci = [get_num_nci(frag, prot) for frag in fragments]
    else:
        num_nci = [1] * len(fragments)

    for curr_mapping in combos:
        indices_map = [atom_mapping.index(indices) for indices in curr_mapping]
        curr_frags = [fragments[i] for i in indices_map if num_nci[i] > 0]
        curr_frags_ev = [frags_ev[i] for i in indices_map if num_nci[i] > 0]

        if len(curr_frags) < 1:
            continue
        ligands.append(_prepare_ligand(
            curr_frags, curr_frags_ev, curr_mapping, rdmol, bond_encoder, frag_embedder,
            ligand_coord, ligand_onehot, frag_adj=not filter_by_nci
        ))
        frag_smiles = [rdmol_to_smiles(frag) for frag in curr_frags_ev]
        smiles.append(('.'.join(frag_smiles)))

    assert ligands, "Ligand cannot be fragmented or has no interacting fragments."

    return ligands, smiles, fragments


def fragment_and_augment(rdmol, min_frag_size):
    # fragments
    fragments, atom_mapping, frags_ev = get_BRICS_fragments(rdmol)

    if min_frag_size > 1 and rdmol.GetNumHeavyAtoms() > min_frag_size:


        frag_indices = [i for i, frag in enumerate(fragments) if frag.GetNumHeavyAtoms() < min_frag_size]
        revised_frags = fragments.copy()
        revised_atom_mapping = atom_mapping.copy()
        revised_frags_ev = frags_ev.copy()
        while frag_indices:
            # check based on atom mapping if fragments were connected before
            revised_frags_ev, revised_atom_mapping = \
                stitch_small_fragments(revised_frags_ev,
                                       revised_atom_mapping,
                                       frag_indices,
                                       rdmol)
            # deduplicate revised_frags_ev
            revised_frags, revised_atom_mapping, revised_frags_ev = \
                get_stitched_frags_wo_ev(revised_frags_ev, revised_atom_mapping)

            frag_indices = [i for i, frag in enumerate(revised_frags) if frag.GetNumHeavyAtoms() < min_frag_size]

        # data augmentation: combine in all ways to full ligand based on atom mapping
        def handler(signum, frame):
            raise TimeoutError("Combining fragments took too long")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60)  # Set the alarm for 60 seconds

        try:
            combos = combine_fragments(revised_atom_mapping)
        finally:
            signal.alarm(0)  # Disable the alarm
        fragments, frags_ev, atom_mapping = revised_frags, revised_frags_ev, revised_atom_mapping
    else:
        combos = [atom_mapping]

    return combos, fragments, frags_ev, atom_mapping


def _prepare_ligand(
        fragments,
        frags_ev,
        atom_mapping,
        rdmol,
        bond_encoder,
        frag_embedder,
        ligand_coord,
        ligand_onehot,
        frag_adj=True,):

    frag_feats = [encode_fragment(frag, frag_ev, frag_embedder) for frag, frag_ev in zip(fragments, frags_ev)]

    smiles = '.'.join([rdmol_to_smiles(frag) for frag in fragments])
    smiles_ev = '.'.join([Chem.MolToSmiles(frag_ev) for frag_ev in frags_ev])

    use_ev = frag_embedder.hparams.get('ligand_encoder_params').get('use_ev', False)
    local_frames = np.array([generate_local_frame_aa(frag_ev, use_ev=use_ev) for frag_ev in frags_ev])
    local_frames = torch.tensor(local_frames, dtype=FLOAT_TYPE)
    axis_angles = so3.rotation_vector_from_matrix(local_frames)

    # Bonds and connection between fragments
    adj = np.ones((rdmol.GetNumAtoms(), rdmol.GetNumAtoms())) * bond_encoder['NOBOND']
    if frag_adj:
        adj_frag = np.ones((len(fragments), len(fragments))) * frag_conn_encoder['NOBOND']
    for b in rdmol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        adj[i, j] = bond_encoder[str(b.GetBondType())]
        adj[j, i] = adj[i, j]  # undirected graph

        if frag_adj:
            # if i and j are in different index groups of atom_mpping -> connect fragments
            frag_i = [idx for idx, indices in enumerate(atom_mapping) if i in indices][0]
            frag_j = [idx for idx, indices in enumerate(atom_mapping) if j in indices][0]
            if frag_i != frag_j:
                adj_frag[frag_i, frag_j] = frag_conn_encoder['SINGLE']
                adj_frag[frag_j, frag_i] = adj_frag[frag_i, frag_j]

    # molecular graph is undirected -> don't save redundant information
    bonds = np.stack(np.triu_indices(len(ligand_coord), k=1), axis=0)
    # bonds = np.stack(np.ones_like(adj).nonzero(), axis=0)
    bond_types = adj[bonds[0], bonds[1]].astype('int64')
    bonds = torch.from_numpy(bonds)
    bond_types = F.one_hot(torch.from_numpy(bond_types), num_classes=len(bond_encoder))

    # frag connections
    frag_connections = np.stack(np.triu_indices(len(fragments), k=1), axis=0)
    if frag_adj:
        frag_conn_types = adj_frag[frag_connections[0], frag_connections[1]].astype('int64')
    else:
        frag_conn_types = np.zeros(frag_connections.shape[1], dtype='int64')
    frag_connections = torch.from_numpy(frag_connections)
    frag_conn_types = F.one_hot(torch.from_numpy(frag_conn_types), num_classes=len(frag_conn_encoder))

    ligand = {
        'x': ligand_coord.to(dtype=FLOAT_TYPE),
        'one_hot': ligand_onehot.to(dtype=FLOAT_TYPE),
        'mask': torch.zeros(len(ligand_coord), dtype=INT_TYPE),
        'bonds': bonds.to(INT_TYPE),
        'bond_one_hot': bond_types.to(FLOAT_TYPE),
        'bond_mask': torch.zeros(bonds.size(1), dtype=INT_TYPE),
        'size': torch.tensor([len(ligand_coord)], dtype=INT_TYPE),
        'n_bonds': torch.tensor([len(bond_types)], dtype=INT_TYPE),
        'coarse_x': torch.stack([f['geom_mean'] for f in frag_feats], dim=0),
        'coarse_one_hot': torch.cat([f['frag_desc'].detach() for f in frag_feats], dim=0), # not actually OHE but same naming scheme for convenience
        'coarse_mask': torch.zeros(len(frag_feats), dtype=INT_TYPE),
        'coarse_bonds': frag_connections.to(INT_TYPE),
        'coarse_bond_one_hot': frag_conn_types.to(FLOAT_TYPE),
        'coarse_bond_mask': torch.zeros(frag_connections.size(1), dtype=INT_TYPE),
        'coarse_size': torch.tensor([len(frag_feats)], dtype=INT_TYPE),
        'coarse_n_bonds': torch.tensor([len(frag_conn_types)], dtype=INT_TYPE),
        'axis_angle': axis_angles,
        'frag_smiles': smiles,
        'frag_smiles_ev': smiles_ev,
    }

    return ligand


def get_side_chain_vectors(res, index_dict, size=None):
    if size is None:
        size = max([x for aa in index_dict.values() for x in aa.values()]) + 1

    resname = three_to_one(res.get_resname())

    out = np.zeros((size, 3))
    for atom in res.get_atoms():
        if atom.get_name() in index_dict[resname]:
            idx = index_dict[resname][atom.get_name()]
            out[idx] = atom.get_coord() - res['CA'].get_coord()
        # else:
        #     if atom.get_name() != 'CA' and not atom.get_name().startswith('H'):
        #         print(resname, atom.get_name())

    return out


def get_normal_modes(res, normal_mode_dict):
    nm = normal_mode_dict[(res.get_parent().id, res.id[1], 'CA')]  # (n_modes, 3)
    return nm


def prepare_pocket(biopython_residues,
                   amino_acid_encoder,
                   residue_bond_encoder,
                   pocket_representation='surface',
                   pdb_file=None,
                   msms_bin=None,
                   msms_resolution=1.0,
                   full_protein_dir=None,
                   ligand_coords=None,
                   crossdocked=True,
                   msms_processed_dir=None,
                   pocket_resids=None,
                   ):

    assert pdb_file is None or pocket_representation in ['CA+', 'surface'], \
        "vector features are only supported for CA+ or surface pockets"

    # sort residues
    biopython_residues = sorted(biopython_residues, key=lambda x: (x.parent.id, x.id[1]))

    if pdb_file is not None and pocket_representation == 'CA+':
        # preprocessed normal mode eigenvectors
        if isinstance(pdb_file, dict):
            nma_dict = pdb_file

        # PDB file
        else:
            nma_dict = pdb_to_normal_modes(str(pdb_file))

    # also for surface for simplifying evaluations
    if pocket_representation in ['CA+', 'surface']:
        ca_coords = np.zeros((len(biopython_residues), 3))
        ca_types = np.zeros(len(biopython_residues), dtype='int64')

        v_dim = max([x for aa in aa_atom_index.values() for x in aa.values()]) + 1
        vec_feats = np.zeros((len(biopython_residues), v_dim, 3), dtype='float32')
        nf_nma = 5
        nma_feats = np.zeros((len(biopython_residues), nf_nma, 3), dtype='float32')

        edges = []  # CA-CA and CA-side_chain
        edge_types = []
        last_res_id = None
        all_atom_coords = []
        all_atom_types = []
        for i, res in enumerate(biopython_residues):
            aa = amino_acid_encoder[three_to_one(res.get_resname())]
            ca_coords[i, :] = res['CA'].get_coord()
            ca_types[i] = aa
            all_atom_res_coords = []
            all_atom_res_types = []
            if pocket_representation == 'surface':
                atoms = res.get_atoms()
                for atom in atoms:
                    if atom.element in protein_atom_mapping:
                        all_atom_res_coords.append(torch.from_numpy(atom.get_coord()))
                        all_atom_res_types.append(protein_atom_mapping[atom.element])
                all_atom_coords.extend(all_atom_res_coords)
                all_atom_types.extend(all_atom_res_types)

            vec_feats[i] = get_side_chain_vectors(res, aa_atom_index, v_dim)
            if pdb_file is not None and pocket_representation == 'CA+':
                nma_feats[i] = get_normal_modes(res, nma_dict)

            # add edges between contiguous CA atoms
            if i > 0 and res.id[1] == last_res_id + 1:
                edges.append((i - 1, i))
                edge_types.append(residue_bond_encoder['CA-CA'])

            last_res_id = res.id[1]

        # Coordinates
        pocket_coords = torch.from_numpy(ca_coords)

        # Features
        pocket_onehot = F.one_hot(torch.from_numpy(ca_types),
                                  num_classes=len(amino_acid_encoder))

        vector_features = torch.from_numpy(vec_feats)
        nma_features = torch.from_numpy(nma_feats)

        # Bonds
        if len(edges) < 1:
            edges = torch.empty(2, 0)
            edge_types = torch.empty(0, len(residue_bond_encoder))
        else:
            edges = torch.tensor(edges).T
            edge_types = F.one_hot(torch.tensor(edge_types),
                                   num_classes=len(residue_bond_encoder))

        if pocket_representation == 'surface':
            all_atom_types_ohe = F.one_hot(torch.tensor(all_atom_types), num_classes=len(protein_atom_mapping))
            all_atom_coords = torch.stack(all_atom_coords)

            if crossdocked:
                parent_fn = ('_').join(pdb_file.stem.split('_')[:3])
                full_pdb_file = full_protein_dir / pdb_file.parent.stem / f'{parent_fn}.pdb'
            else:
                full_pdb_file = pdb_file
            assert full_pdb_file.exists(), f"Full protein PDB file not found: {full_pdb_file}"
            # assert that file not empty
            assert full_pdb_file.stat().st_size > 0, f"Protonated full protein PDB file is empty: {full_pdb_file}"
            complete_structure = PDBParser(QUIET=True).get_structure('', full_pdb_file)[0]
            if msms_processed_dir is not None:
                msms_processed_fn = msms_processed_dir / f'{pdb_file.stem}.npy'
                if msms_processed_fn.exists():
                    msms_out = np.load(msms_processed_fn, allow_pickle=True)
                else:
                    msms_out = pdb_to_points_normals(
                            complete_structure, msms_bin, msms_resolution,
                            filter=lambda a: default_filter(a.parent), return_faces=False)
                    np.save(msms_processed_fn, msms_out)
            else:
                msms_out = pdb_to_points_normals(
                        complete_structure, msms_bin, msms_resolution,
                        filter=lambda a: default_filter(a.parent), return_faces=False)

            surface_coords = torch.from_numpy(msms_out[0])
            surface_norms = torch.from_numpy(msms_out[1])

            # keep only surface points closest to ligand
            if ligand_coords is not None:
                dists = torch.cdist(ligand_coords, surface_coords.float())
                mask = dists.min(0).values < 7.0
                surface_coords = surface_coords[mask]
                surface_norms = surface_norms[mask]
                # remove points with norms pointing away from the ligand
                surface_to_ligand = ligand_coords.mean(0) - surface_coords
                dot = (surface_to_ligand * surface_norms).sum(-1)
                mask = dot > 0
                surface_coords = surface_coords[mask]
                surface_norms = surface_norms[mask]

                if len(surface_coords) < 250:
                    raise ValueError(f"Too few surface points: {len(surface_coords)}")
            elif pocket_resids is not None:
                active_coords = []
                for res_id in pocket_resids:
                    res = complete_structure[(' ', res_id, ' ')]
                    for atom in res:
                        active_coords.append(atom.get_coord())
                active_coords = torch.tensor(np.array(active_coords))
                dists = torch.cdist(active_coords, surface_coords.float())
                mask = dists.min(0).values < 3.0
                surface_coords = surface_coords[mask]
                surface_norms = surface_norms[mask]
                if len(surface_coords) < 250:
                    raise ValueError(f"Too few surface points: {len(surface_coords)}")


    else:
        raise NotImplementedError(
            f"Pocket representation '{pocket_representation}' not implemented")

    # pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in biopython_residues]

    pocket = {
        'x': pocket_coords.to(dtype=FLOAT_TYPE),
        'one_hot': pocket_onehot.to(dtype=FLOAT_TYPE),
        # 'ids': pocket_ids,
        'size': torch.tensor([len(pocket_coords)], dtype=INT_TYPE),
        'mask': torch.zeros(len(pocket_coords), dtype=INT_TYPE),
        'bonds': edges.to(INT_TYPE),
        'bond_one_hot': edge_types.to(FLOAT_TYPE),
        'bond_mask': torch.zeros(edges.size(1), dtype=INT_TYPE),
        'n_bonds': torch.tensor([len(edge_types)], dtype=INT_TYPE),
    }

    if vector_features is not None:
        pocket['v'] = vector_features.to(dtype=FLOAT_TYPE)

    if pdb_file is not None and pocket_representation == 'CA+':
        pocket['nma_vec'] = nma_features.to(dtype=FLOAT_TYPE)

    if pocket_representation == 'surface':
        pocket['normals_surface'] = surface_norms.to(dtype=FLOAT_TYPE)
        pocket['x_surface'] = surface_coords.to(dtype=FLOAT_TYPE)
        pocket['size_surface'] = torch.tensor([len(surface_coords)], dtype=INT_TYPE)
        pocket['mask_surface'] = torch.zeros(len(surface_coords), dtype=INT_TYPE)
        pocket['atom_xyz_surface'] = all_atom_coords.to(dtype=FLOAT_TYPE)
        pocket['atomtypes_surface'] = all_atom_types_ohe.to(dtype=FLOAT_TYPE)
        pocket['atom_batch_surface'] = torch.zeros(len(all_atom_coords), dtype=INT_TYPE)
        # pocket['edges'] = surface_edges.to(INT_TYPE)

        assert len(pocket['x_surface']) > 0, f"Empty surface pocket."

    return pocket, biopython_residues


def process_raw_pair(biopython_model,
                     rdmol,
                     frag_embedder,
                     dist_cutoff=None,
                     pocket_representation='surface',
                     pdb_file=None,
                     return_pocket_pdb=False,
                     min_frag_size=1,
                     msms_bin=None,
                     msms_resolution=1.0,
                     full_protein_dir=None,
                     crossdocked=True,
                     msms_processed_dir=None,
                     filter_by_nci=False):

    # Process ligand
    # remove H atoms if not in atom_encoder
    if 'H' not in atom_encoder:
        rdmol = Chem.RemoveAllHs(rdmol, sanitize=False)

    ligand, frag_smiles, fragments = prepare_ligand(rdmol,
                                                    atom_encoder,
                                                    bond_encoder,
                                                    frag_embedder,
                                                    min_frag_size,
                                                    filter_by_nci,
                                                    pdb_file)

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in biopython_model.get_residues():

        # Remove non-standard amino acids and HETATMs
        if not is_aa(residue.get_resname(), standard=True):
            continue

        res_coords = torch.from_numpy(np.array([a.get_coord() for a in residue.get_atoms()]))
        if dist_cutoff is None or (((res_coords[:, None, :] - ligand[0]['x'][None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    pocket, pocket_residues = prepare_pocket(
        pocket_residues,
        aa_encoder,
        residue_bond_encoder,
        pocket_representation,
        pdb_file,
        msms_bin=msms_bin,
        msms_resolution=msms_resolution,
        full_protein_dir=full_protein_dir,
        ligand_coords=ligand[0]['x'],
        crossdocked=crossdocked,
        msms_processed_dir=msms_processed_dir,
    )

    if return_pocket_pdb:
        builder = StructureBuilder.StructureBuilder()
        builder.init_structure("")
        builder.init_model(0)
        pocket_struct = builder.get_structure()
        for residue in pocket_residues:
            chain = residue.get_parent().get_id()

            # init chain if necessary
            if not pocket_struct[0].has_id(chain):
                builder.init_chain(chain)

            # add residue
            pocket_struct[0][chain].add(residue)

        pocket['pocket_pdb'] = pocket_struct
    # if return_pocket_pdb:
    #     pocket['residues'] = [prepare_internal_coord(res) for res in pocket_residues]

    return ligand, pocket, frag_smiles, fragments


def process_pocket_only(pdb_file,
                        msms_bin,
                        msms_resolution=1.0,
                        pocket_resids=None,
                        ligand_coords=None,
                        dist_cutoff=8.0,):
    pdb_model = PDBParser(QUIET=True).get_structure('', pdb_file)[0]

    if pocket_resids is not None:
        active_coords = []
        for chain in pdb_model:
            for res_id in pocket_resids:
                res = chain[(' ', res_id, ' ')]
                for atom in res:
                    active_coords.append(atom.get_coord())
        active_coords = torch.tensor(np.array(active_coords))

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_model.get_residues():

        # Remove non-standard amino acids and HETATMs
        if not is_aa(residue.get_resname(), standard=True):
            continue

        res_coords = torch.from_numpy(np.array([a.get_coord() for a in residue.get_atoms()]))
        if ligand_coords is not None:
            if dist_cutoff is None or (((res_coords[:, None, :] - ligand_coords) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
                pocket_residues.append(residue)
        elif active_coords is not None:
            if  (((res_coords[:, None, :] - active_coords) ** 2).sum(-1) ** 0.5).min() < 1.0:
                pocket_residues.append(residue)

    pocket, pocket_residues = prepare_pocket(
        pocket_residues,
        aa_encoder,
        residue_bond_encoder,
        'surface',
        pdb_file,
        msms_bin=msms_bin,
        msms_resolution=msms_resolution,
        ligand_coords=ligand_coords,
        crossdocked=False,
    )

    return pocket


def rdmol_to_smiles(rdmol):
    mol = Chem.Mol(rdmol)
    # Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def get_n_nodes(lig_positions, pocket_positions, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    n_nodes_lig = [len(x) for x in lig_positions]
    n_nodes_pocket = [len(x) for x in pocket_positions]

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def get_type_histogram(one_hot, type_encoder):

    one_hot = np.concatenate(one_hot, axis=0)

    decoder = list(type_encoder.keys())
    counts = {k: 0 for k in type_encoder.keys()}
    for a in [decoder[x] for x in one_hot.argmax(1)]:
        counts[a] += 1

    return counts


def get_bit_histogram(bit_vectors):
    # get counts based on each bit position (the times there is a one at index 0 etc.)
    bit_vectors = np.concatenate(bit_vectors, axis=0)
    bit_vectors = bit_vectors.reshape(-1, bit_vectors.shape[-1])

    counts = bit_vectors.sum(0)

    return {i: counts[i] for i in range(len(counts))}


def get_distribution(embeddings):
    embeddings = np.concatenate(embeddings, axis=0)

    means = embeddings.mean(axis=0)
    std_devs = embeddings.std(axis=0)

    return means, std_devs


def center_data(ligand, pocket, pocket_representation = 'CA+'):
    if pocket_representation == 'surface':
        pocket_com = scatter_mean(pocket['x_surface'], pocket['mask_surface'], dim=0)
        pocket['x_surface'] = pocket['x_surface'] - pocket_com[pocket['mask_surface']]
    else:
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
    pocket['x'] = pocket['x'] - pocket_com[pocket['mask']]
    ligand['x'] = ligand['x'] - pocket_com[ligand['mask']]
    if 'coarse_x' in ligand:
        ligand['coarse_x'] = ligand['coarse_x'] - pocket_com[ligand['coarse_mask']]
    return ligand, pocket


def collate_entity(batch):

    out = {}
    for prop in batch[0].keys():

        if prop in {'name', 'frag_smiles', 'frag_smiles_ev'}:
            out[prop] = [x[prop] for x in batch]

        elif prop in {'size', 'n_bonds', 'coarse_size', 'coarse_n_bonds', 'size_surface'} :
            out[prop] = torch.tensor([x[prop] for x in batch])

        elif prop == 'bonds':
            # index offset
            offset = list(accumulate([x['size'] for x in batch]))
            offset.insert(0, 0)
            out[prop] = torch.cat([x[prop] + offset[i] for i, x in enumerate(batch)], dim=1)

        elif prop == 'coarse_bonds':
            # index offset
            offset = list(accumulate([x['coarse_size'] for x in batch]))
            offset.insert(0, 0)
            out[prop] = torch.cat([x[prop] + offset[i] for i, x in enumerate(batch)], dim=1)

        elif prop == 'residues':
            out[prop] = list(chain.from_iterable(x[prop] for x in batch))

        elif prop in {'mask', 'bond_mask', 'coarse_mask', 'coarse_bond_mask',
                      'mask_surface', 'atom_batch_surface'}:
            pass  # batch masks will be written later

        else:
            out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        # Create batch masks
        # make sure indices in batch start at zero (needed for torch_scatter)
        if prop == 'x':
            out['mask'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                     for i, x in enumerate(batch)], dim=0)
        if prop == 'coarse_x':
            out['coarse_mask'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                     for i, x in enumerate(batch)], dim=0)

        if prop == 'x_surface':
            out['mask_surface'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                     for i, x in enumerate(batch)], dim=0)

        if prop == 'atom_xyz_surface':
            out['atom_batch_surface'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                     for i, x in enumerate(batch)], dim=0)

        if prop == 'bond_one_hot':
            # TODO: this is not necessary as it can be computed on-the-fly as bond_mask = mask[bonds[0]] or bond_mask = mask[bonds[1]]
            out['bond_mask'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                          for i, x in enumerate(batch)], dim=0)

        if prop == 'coarse_bond_one_hot':
            out['coarse_bond_mask'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                          for i, x in enumerate(batch)], dim=0)

    return out


def split_entity(batch):
    """ Splits a batch into items and returns a list. """

    batch_size = len(torch.unique(batch['mask']))
    if 'mask_surface' in batch:
        bs_surface = len(torch.unique(batch['mask_surface']))
        assert batch_size == bs_surface, f"Batch size mismatch: {batch_size} vs {bs_surface}"

    out = {}
    for prop in batch.keys():

        if prop in {'name', 'size', 'n_bonds', 'coarse_size', 'coarse_n_bonds', 'size_surface', 'frag_smiles', 'frag_smiles_ev'}:
            out[prop] = batch[prop]  # already a list

        elif prop == 'bonds':
            offsets = list(accumulate(batch['size'][:-1]))
            offsets.insert(0, 0)
            out[prop] = batch_to_list_for_indices(batch[prop], batch['bond_mask'], offsets)

        elif prop == 'coarse_bonds':
            offsets = list(accumulate(batch['coarse_size'][:-1]))
            offsets.insert(0, 0)
            # remove offset elements when there is only one fragment in a data point
            offsets = [offset for offset, size in zip(offsets, batch['coarse_size']) if size > 1]
            out[prop] = batch_to_list_for_indices(batch[prop], batch['coarse_bond_mask'], offsets)
            for i, size in enumerate(batch['coarse_size']):
                if size == 1:
                    out[prop].insert(i, torch.tensor([], dtype=torch.int64, device=batch[prop].device))

        elif prop in {'bond_one_hot', 'bond_mask'}:
            out[prop] = batch_to_list(batch[prop], batch['bond_mask'])

        elif prop in {'coarse_bond_one_hot', 'coarse_bond_mask'}:
            out[prop] = batch_to_list(batch[prop], batch['coarse_bond_mask'])
            for i, size in enumerate(batch['coarse_size']):
                if size == 1:
                    out[prop].insert(i, torch.tensor([], dtype=torch.float32, device=batch[prop].device))

        elif prop in {'x', 'one_hot'}:
            out[prop] = batch_to_list(batch[prop], batch['mask'])

        elif prop in {'x_surface', 'normals_surface'}:
            out[prop] = batch_to_list(batch[prop], batch['mask_surface'])

        elif prop in {'coarse_x', 'coarse_one_hot', 'coarse_mask', 'axis_angle'}:
            out[prop] = batch_to_list(batch[prop], batch['coarse_mask'])

        elif prop == 'mask_surface':
            out[prop] = batch_to_list(batch[prop], batch['mask_surface'])

        elif prop == 'atom_xyz_surface':
            out[prop] = batch_to_list(batch[prop], batch['atom_batch_surface'])

        elif prop == 'atomtypes_surface':
            out[prop] = batch_to_list(batch[prop], batch['atom_batch_surface'])

        elif prop == 'atom_batch_surface':
            out[prop] = batch_to_list(batch[prop], batch['atom_batch_surface'])

        else:
            out[prop] = batch_to_list(batch[prop], batch['mask'])

    out = [{k: v[i] for k, v in out.items()} for i in range(batch_size)]
    return out


def repeat_items(batch, repeats):
    batch_list = split_entity(batch)
    out = collate_entity([x for _ in range(repeats) for x in batch_list])
    return out
