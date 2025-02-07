import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO
import prody


INTERACTION_TYPES = {
        'hydrophobic_interactions': 'hydrophobic_interaction',
        'hydrogen_bonds': 'hydrogen_bond',
        'water_bridges': 'water_bridge',
        'salt_bridges': 'salt_bridge',
        'pi_stacks': 'pi_stack',
        'pi_cation_interactions': 'pi_cation_interaction',
        'halogen_bonds': 'halogen_bond',
        'metal_complexes': 'metal_complex',
}

INTERACTION_LIST = list(INTERACTION_TYPES.keys())

REDUCE_EXEC = 'reduce'


def prepare_protein(protein_path, outdir, verbose=False, reduce_exec=REDUCE_EXEC):
    structure = prody.parsePDB(protein_path).select('protein')
    hydrogens = structure.select('hydrogen')
    if hydrogens is None or len(hydrogens) < len(set(structure.getResnums())):
        if verbose:
            print('Target structure is not protonated. Adding hydrogens...')

        output_path = outdir / 'protonated.pdb'

        reduce_result = subprocess.run(f'{reduce_exec} {protein_path} > {output_path} 2> /dev/null', shell=True, capture_output=True, text=True)
        if reduce_result.returncode != 0:
            raise RuntimeError('Error during reduce execution:', reduce_result.stderr)

    else:
        output_path = Path(protein_path)

    return output_path


def prepare_mol(mol, outdir):
    mol_pdb_path = outdir / 'mol.pdb'
    Chem.MolToPDBFile(mol, str(mol_pdb_path))
    return mol_pdb_path


def combine_mol_pdb(mol_pdb, protein_pdb, outfile):
    parser = PDBParser(QUIET=True)

    # Load the first PDB file
    structure1 = parser.get_structure('1', mol_pdb)

    # Load the second PDB file
    structure2 = parser.get_structure('2', protein_pdb)

    # Create a new model to store the combined structure
    model1 = structure1[0]

    # Iterate through chains in the second structure
    for model in structure2:
        for chain in model:
            # Create a new chain with a unique identifier
            new_chain = chain.copy()
            new_chain.id = chr(ord(chain.id) + len(model1.child_list))

            # Add the new chain to the first model
            model1.add(new_chain)

    # Create a PDBIO to write the merged structure to a file
    io = PDBIO()
    io.set_structure(structure1)
    io.save(outfile)


def run_plip(structure, outdir, plip_exec='plipcmd'):
    plip_out = outdir / 'plip_out'
    command = f'{plip_exec} -f {structure} -o {plip_out} -x'
    subprocess.run(command, shell=True)
    return plip_out


def read_plip(plip_path):
    tree = ET.parse(plip_path)
    root = tree.getroot()

    binding_site = root.findall('bindingsite')[0]
    interactions = binding_site.find('interactions')

    results = {}
    for inter_type, tag in INTERACTION_TYPES.items():
        inter = interactions.find(inter_type).findall(tag)

        results[inter_type] = len(inter)

    return results
