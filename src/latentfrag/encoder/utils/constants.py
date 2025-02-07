######## PROTEIN ########
protein_atom_mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "Se": 5}

######## LIGAND ########
ligand_atom_mapping = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'NH': 10, 'N+': 11, 'O-': 12,}
ligand_atom_mapping_ev = ligand_atom_mapping.copy()
ligand_atom_mapping_ev.update({'R1': 13, 'R2': 14, 'R3': 15, 'R4': 16, 'R5': 17, 'R6': 18, 'R8': 19, 'R9': 20, 'R10': 21, 'R11': 22, 'R12': 23, 'R13': 24, 'R14': 25, 'R15': 26, 'R16': 27})

possible_atom_charges = list(range(-1, 4))
ligand_charge_mapping = {x: i for i, x in enumerate(possible_atom_charges)}

possible_atom_degrees = list(range(7))
ligand_degree_mapping = {x: i for i, x in enumerate(possible_atom_degrees)}

ligand_hybridization_mapping = {"SP": 0, "SP2": 1, "SP3": 2, "SP2D": 3, "SP3D": 4, "SP3D2": 5}

ligand_chiral_tag_mapping = {"CHI_UNSPECIFIED": 0, "CHI_TETRAHEDRAL_CW": 1, "CHI_TETRAHEDRAL_CCW": 2, "CHI_OTHER": 3}

bond_mapping = {"NOBOND": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}

possible_ring_sizes = list(range(3, 19))
possible_ring_sizes.append(0)  # no ring
possible_ring_sizes.append(1)  # other ring size
ring_size_mapping = {x: i for i, x in enumerate(possible_ring_sizes)}

# min and max limits for normalization (from fragment train set)
atom_num_lims = (1, 20)
bond_num_lims = (0, 24)
ring_num_lims = (0, 5)
mw_lims = (16.0, 716.0)
logp_lims = (-7.8, 8.5)
tpsa_lims = (0.0, 233.0)
hbd_lims = (0, 10)
hba_lims = (0, 13)

######## OTHER STUFF ########

# non-covalent interactions with proteins
NCIS = {
        'hydrophobic_interactions': 'hydrophobic_interaction',
        'hydrogen_bonds': 'hydrogen_bond',
        'water_bridges': 'water_bridge',
        'salt_bridges': 'salt_bridge',
        'pi_stacks': 'pi_stack',
        'pi_cation_interactions': 'pi_cation_interaction',
        'halogen_bonds': 'halogen_bond',
}
