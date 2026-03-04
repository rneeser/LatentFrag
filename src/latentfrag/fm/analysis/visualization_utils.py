
import warnings

import torch
from rdkit import Chem
try:
    from Bio.PDB.Polypeptide import one_to_three
except ImportError:
    from Bio.PDB.Polypeptide import protein_letters_3to1
    _one_to_three = {v: k for k, v in protein_letters_3to1.items()}
    def one_to_three(s):
        return _one_to_three[s]

from latentfrag.fm.data.molecule_builder import build_molecule


def mols_to_pdbfile(rdmols, filename, flavor=0):
    pdb_str = ""
    for i, mol in enumerate(rdmols):
        pdb_str += f"MODEL{i + 1:>9}\n"
        block = Chem.MolToPDBBlock(mol, flavor=flavor)
        block = "\n".join(block.split("\n")[:-2])  # remove END
        pdb_str += block + "\n"
        pdb_str += f"ENDMDL\n"
    pdb_str += f"END\n"

    with open(filename, 'w') as f:
        f.write(pdb_str)

    return pdb_str


def pocket_to_rdkit(pocket, pocket_representation, atom_encoder=None,
                    atom_decoder=None, aa_decoder=None, aa_atom_index=None):

    # Move all tensors to CPU for RDKit/PDB construction
    pocket = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in pocket.items()}

    rdpockets = []
    for i in torch.unique(pocket['mask']):

        node_coord = pocket['x'][pocket['mask'] == i]
        h = pocket['one_hot'][pocket['mask'] == i]

        pdb_infos = []

        if pocket_representation == 'CA+' or pocket_representation == 'surface':
            aa_types = [aa_decoder[b] for b in h.argmax(-1)]
            side_chain_vec = pocket['v'][pocket['mask'] == i]

            coord = []
            atom_types = []
            for resi, (xyz, aa, vec) in enumerate(zip(node_coord, aa_types, side_chain_vec)):

                # CA not treated differently with updated atom dictionary
                for atom_name, idx in aa_atom_index[aa].items():

                    coord.append(xyz + vec[idx])
                    atom_types.append(atom_name[0])

                    info = Chem.AtomPDBResidueInfo()
                    # info.SetChainId('A')
                    info.SetResidueName(one_to_three(aa))
                    info.SetResidueNumber(resi + 1)
                    info.SetOccupancy(1.0)
                    info.SetTempFactor(0.0)
                    info.SetName(f' {atom_name:<3}')
                    pdb_infos.append(info)

            coord = torch.stack(coord, dim=0)

        else:
            raise NotImplementedError(f"{pocket_representation} residue representation not supported")

        atom_types = torch.tensor([atom_encoder[a] for a in atom_types])
        rdmol = build_molecule(coord, atom_types, atom_decoder=atom_decoder)

        if len(pdb_infos) == len(rdmol.GetAtoms()):
            for a, info in zip(rdmol.GetAtoms(), pdb_infos):
                a.SetPDBResidueInfo(info)

        rdpockets.append(rdmol)

    return rdpockets
