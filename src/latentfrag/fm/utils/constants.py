import torch
from rdkit import Chem

FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

# ------------------------------------------------------------------------------
# Full ligand constants
# ------------------------------------------------------------------------------
atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'NH': 10, 'N+': 11, 'O-': 12, 'NOATOM': 13}
atom_decoder = ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'NH', 'N+', 'O-', 'NOATOM']

bond_encoder = {"NOBOND": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, 'AROMATIC': 4}
bond_decoder = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

# ------------------------------------------------------------------------------
# Fragment constants
# ------------------------------------------------------------------------------
frag_conn_encoder = {"NOBOND": 0, "SINGLE": 1}
frag_conn_decoder = [None, Chem.rdchem.BondType.SINGLE]


# ------------------------------------------------------------------------------
# Protein constants
# ------------------------------------------------------------------------------
aa_encoder = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
aa_decoder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

residue_bond_encoder = {'CA-CA': 0, 'CA-SS': 1, 'NOBOND': 2}
residue_bond_decoder = ['CA-CA', 'CA-SS', None]

protein_atom_mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "Se": 5}

aa_atom_index = {
    'A': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4},
    'C': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'SG': 5},
    'D': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'OD2': 7},
    'E': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'OE1': 7, 'OE2': 8},
    'F': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9, 'CZ': 10},
    'G': {'N': 0, 'CA': 1, 'C': 2, 'O': 3},
    'H': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'ND1': 6, 'CD2': 7, 'CE1': 8, 'NE2': 9},
    'I': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6, 'CD1': 7},
    'K': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'CE': 7, 'NZ': 8},
    'L': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7},
    'M': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'SD': 6, 'CE': 7},
    'N': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'ND2': 7},
    'P': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6},
    'Q': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'OE1': 7, 'NE2': 8},
    'R': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'NE': 7, 'CZ': 8, 'NH1': 9, 'NH2': 10},
    'S': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG': 5},
    'T': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG1': 5, 'CG2': 6},
    'V': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6},
    'W': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'NE1': 8, 'CE2': 9, 'CE3': 10, 'CZ2': 11, 'CZ3': 12, 'CH2': 13},
    'Y': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9, 'CZ': 10, 'OH': 11},
}

vdw_radii_msms = {'N': 1.55, 'O': 1.52, 'C': 1.70, 'H': 1.10, 'S': 1.80, 'P': 1.80,
             'SE': 1.90, 'K': 2.75, 'NA': 2.27, 'MG': 1.73, 'ZN': 1.39}  # [A]