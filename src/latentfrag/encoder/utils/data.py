from typing import Callable, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from latentfrag.encoder.utils.constants import (
    protein_atom_mapping,
    ligand_atom_mapping,
    ligand_atom_mapping_ev,
    bond_mapping,
    ligand_charge_mapping,
    ligand_degree_mapping,
    ligand_hybridization_mapping,
    ligand_chiral_tag_mapping,
    ring_size_mapping,
    atom_num_lims,
    bond_num_lims,
    ring_num_lims,
    mw_lims,
    logp_lims,
    tpsa_lims,
    hbd_lims,
    hba_lims,
    )


class TensorDict(dict):
    def __init__(self, **kwargs):
        super(TensorDict, self).__init__(**kwargs)

    def to(self, device):
        for k, v in self.items():
            # if torch.is_tensor(v):
            if hasattr(v, 'to'):
                self[k] = v.to(device)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')


def pdb_filter(struct, model_filter=lambda m: True, chain_filter=lambda c: True,
               residue_filter=lambda r: True, atom_filter=lambda a: True):

    for model in list(struct.get_models()):
        if model_filter(model):
            for chain in list(model.get_chains()):
                if chain_filter(chain):
                    for res in list(chain.get_residues()):
                        if residue_filter(res):
                            for atom in list(res.get_atoms()):
                                if not atom_filter(atom):
                                    res.detach_child(atom.id)
                        else:
                            chain.detach_child(res.id)
                else:
                    model.detach_child(chain.id)
        else:
            struct.detach_child(model.id)

    return struct


def remove_hetatm(infile, outfile):

    struct = PDBParser(QUIET=True).get_structure('', infile)
    struct = pdb_filter(struct, model_filter=lambda m: m.id == 0,
                        residue_filter=is_aa)
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(outfile))


def default_filter(res):
    return is_aa(res)


def prepare_pdb(pdb: Union[Path, Model, Chain],
                filter: Callable = default_filter,
                name: str = '') -> TensorDict:

    if isinstance(pdb, Path):
        # Read the input structure
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("pdb-structure", str(pdb))
        model = struct[0]  # in case there are several models
    else:
        model = pdb

    # Process atoms
    residues = [res for res in model.get_residues() if filter(res)]
    num_residues = len(residues)
    atoms = [a for res in residues for a in res.get_atoms()]

    coords = []
    types = []
    res_ids = []
    # res_batch = []
    # curr_batch = 0
    for atom in atoms:
        if atom.element in protein_atom_mapping:
            coords.append(torch.from_numpy(atom.get_coord()))
            types.append(protein_atom_mapping[atom.element])
            res_ids.append(atom.get_parent().id[1]) # some IDs get skipped (correct mapping?)
            # if len(res_ids) > 1 and res_ids[-1] != res_ids[-2]:
            #     curr_batch += 1
                # if all_res_ids[all_res_ids.index(res_ids[-1])-1] != res_ids[-2]:
                #     curr_batch += 1
            # res_batch.append(curr_batch)

    first_res = res_ids[0]
    res_ids_shifted = [x - first_res for x in res_ids]

    coords = torch.stack(coords)
    types_array = torch.zeros((len(types), len(protein_atom_mapping)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Create output dictionary
    protein = {
        "name": name,
        "atom_xyz": coords,
        "atomtypes": types_array,
        "batch_atoms": torch.zeros(len(coords), dtype=int),
        "residue_ids": torch.tensor(res_ids_shifted, dtype=torch.long),
        "seq_length": torch.tensor([num_residues], dtype=torch.long),
    }
    # return protein
    return TensorDict(**protein)

def atom_featurizer(mol, atom_mapping=None):
    # element types
    if atom_mapping is None:
        atom_mapping = ligand_atom_mapping
    types = F.one_hot(
        torch.tensor([encode_atom(a, atom_mapping) for a in mol.GetAtoms()]),
        num_classes=len(atom_mapping)
    )

    # atom belonging in
    in_ring = [x.IsInRing() for x in mol.GetAtoms()]
    in_ring = torch.tensor(in_ring).float().unsqueeze(1)
    is_aromatic = [x.GetIsAromatic() for x in mol.GetAtoms()]
    is_aromatic = torch.tensor(is_aromatic).float().unsqueeze(1)

    # charges
    charges = [ligand_charge_mapping[x.GetFormalCharge()] for x in mol.GetAtoms()]
    charges = F.one_hot(torch.tensor(charges), num_classes=len(ligand_charge_mapping))

    # atom degree
    degrees = [ligand_degree_mapping[x.GetDegree()] for x in mol.GetAtoms()]
    degrees = F.one_hot(torch.tensor(degrees), num_classes=len(ligand_degree_mapping))

    # hybridization state
    if 'R1' in atom_mapping:
        ligand_hybridization_mapping.update({'UNSPECIFIED': 6})
    hybridization = [ligand_hybridization_mapping[str(x.GetHybridization())] for x in mol.GetAtoms()]
    hybridization = F.one_hot(torch.tensor(hybridization), num_classes=len(ligand_hybridization_mapping))

    # chiral tag
    chiral_tag_list = [str(x.GetChiralTag()) for x in mol.GetAtoms()]
    # if not in ligand_chiral_tag_mapping: set to CHI_OTHER
    chiral_tag = [ligand_chiral_tag_mapping.get(x, ligand_chiral_tag_mapping['CHI_OTHER']) for x in chiral_tag_list]
    chiral_tag = F.one_hot(torch.tensor(chiral_tag), num_classes=len(ligand_chiral_tag_mapping))

    atom_feats = torch.cat([types,
                            in_ring,
                            is_aromatic,
                            charges,
                            degrees,
                            hybridization,
                            chiral_tag,
                            ],
                            dim=1)

    return atom_feats


def mol_featurizer(mol):
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        print('WARNING: empty molecule')
    num_atoms = min_max_normalize(num_atoms, atom_num_lims[0], atom_num_lims[1])
    num_bonds = mol.GetNumBonds()
    num_bonds = min_max_normalize(num_bonds, bond_num_lims[0], bond_num_lims[1])
    num_rings = mol.GetRingInfo().NumRings()
    num_rings = min_max_normalize(num_rings, ring_num_lims[0], ring_num_lims[1])
    num_aromatic_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    num_aromatic_rings = min_max_normalize(num_aromatic_rings, ring_num_lims[0], ring_num_lims[1])
    MW = Descriptors.MolWt(mol)
    MW = min_max_normalize(MW, mw_lims[0], mw_lims[1])
    logP = Chem.Crippen.MolLogP(mol)
    logP = min_max_normalize(logP, logp_lims[0], logp_lims[1])
    TPSA = Chem.rdMolDescriptors.CalcTPSA(mol)
    TPSA = min_max_normalize(TPSA, tpsa_lims[0], tpsa_lims[1])
    num_HBD = Chem.rdMolDescriptors.CalcNumHBD(mol)
    num_HBD = min_max_normalize(num_HBD, hbd_lims[0], hbd_lims[1])
    num_HBA = Chem.rdMolDescriptors.CalcNumHBA(mol)
    num_HBA = min_max_normalize(num_HBA, hba_lims[0], hba_lims[1])
    ring_sizes = list(set([len(x) for x in mol.GetRingInfo().AtomRings()]))
    ring_sizes = [ring_size_mapping[x] if x in ring_size_mapping else ring_size_mapping[1] for x in ring_sizes]
    if not ring_sizes:
        ring_sizes = [ring_size_mapping[0]]
    ring_sizes = F.one_hot(torch.tensor(ring_sizes), num_classes=len(ring_size_mapping)).sum(dim=0)

    global_feats = torch.tensor([num_atoms, num_bonds, num_rings, num_aromatic_rings, MW, logP, TPSA, num_HBD, num_HBA])
    global_feats = torch.cat([global_feats, ring_sizes], dim=0)

    return global_feats.float()


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

    isotope = rd_atom.GetIsotope()
    if isotope != 0:
        if element == '*':
            element = 'R'
        if f'{element}{isotope}' in atom_encoder:
            return atom_encoder[f'{element}{isotope}']
        else:
            raise ValueError(f'Unknown exit vector {element}{isotope}')

    return atom_encoder[element]


def featurize_ligand(rdmol, use_ev=False):
    if use_ev:
        atom_mapping = ligand_atom_mapping_ev
    else:
        atom_mapping = ligand_atom_mapping
    if 'H' not in atom_mapping:
        # does not really make a difference as far as I can tell
        rdmol = Chem.RemoveAllHs(rdmol, sanitize=True)

    coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()

    # atom features (dim = 10 + 1 + 1 + 5 + 7 + 6 + 4 = 34)
    atom_feats = atom_featurizer(rdmol, atom_mapping)

    if rdmol.GetNumBonds() > 0:
        bonds_src = torch.tensor([x.GetBeginAtomIdx() for x in rdmol.GetBonds()])
        bonds_dst = torch.tensor([x.GetEndAtomIdx() for x in rdmol.GetBonds()])
        bonds_types = [bond_mapping[str(x.GetBondType())] for x in rdmol.GetBonds()]
        bonds_types = bonds_types + bonds_types
        bonds_types = F.one_hot(torch.tensor(bonds_types), num_classes=len(bond_mapping))
        bonds_types = bonds_types.float()
    else:
        bonds_src = torch.tensor([])
        bonds_dst = torch.tensor([])
        bonds_types = torch.empty(0, len(bond_mapping)) # None

    global_feats = mol_featurizer(rdmol)

    atom_feats = atom_feats.float()

    return {
        'xyz': coords,
        'types': atom_feats,
        'bonds': torch.cat([torch.stack([bonds_src, bonds_dst]).long(),
                            torch.stack([bonds_dst, bonds_src]).long()], dim=1),
        'bond_types': bonds_types,
        'mol_feats': global_feats.expand(1, global_feats.shape[0]),
    }


def prepare_sdf(sdf: Union[Path, Chem.Mol]) -> TensorDict:
    if isinstance(sdf, Path):
        # Read the input structure
        rdmol = Chem.SDMolSupplier(str(sdf), sanitize=True)[0]
    else:
        rdmol = sdf

    if rdmol.GetNumConformers() == 0:
        rdmol = generate_conformers(rdmol, num_confs=1)

    return TensorDict(**featurize_ligand(rdmol))


def generate_conformers(mol, num_confs=10):
    mol = Chem.AddHs(mol)
    embed = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, enforceChirality=True)
    if len(embed) == 0:
        return None
    energies = []
    for conf in mol.GetConformers():
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        ff.Minimize()
        energies.append(ff.CalcEnergy())
    min_index = energies.index(min(energies))
    # remove all other conformers
    for i in range(len(energies)):
        if i != min_index:
            mol.RemoveConformer(i)
    return mol


def min_max_normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def deduplicate_and_combine(coordinates, features):
    # Use unique to get unique coordinates
    unique_coords, inverse_indices = torch.unique(coordinates, dim=0, return_inverse=True)

    # Create a new tensor to hold the combined features
    combined_features = torch.zeros((unique_coords.shape[0], features.shape[1]), dtype=features.dtype)

    # Iterate through unique coordinates and combine their features
    for i in range(len(unique_coords)):
        mask = (inverse_indices == i)
        combined_features[i] = torch.max(features[mask], dim=0)[0]

    return unique_coords, combined_features


def find_nearby_points_and_merge(deduped_coords, combined_features, protein_coords, distance_threshold):
    distances = torch.cdist(deduped_coords, protein_coords)

    # indices of protein coordinates within the distance threshold
    deduped_indices, protein_indices = torch.where(distances <= distance_threshold)

    unique_protein_indices, inverse_indices = torch.unique(protein_indices, return_inverse=True)
    merged_features = torch.zeros((len(unique_protein_indices), combined_features.shape[1]),
                                  dtype=combined_features.dtype, device=combined_features.device)

    # Merge features
    merged_features.index_add_(0, inverse_indices, combined_features[deduped_indices])

    # Ensure binary features
    merged_features.clamp_(max=1)

    return merged_features, unique_protein_indices
