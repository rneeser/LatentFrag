import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
import subprocess

from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO
import pandas as pd
import torch


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--sdf_file', type=str, required=True)
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--pdb_dir', type=str, required=True)

interaction_types = {
        'hydrophobic_interactions': 'hydrophobic_interaction',
        'hydrogen_bonds': 'hydrogen_bond',
        'water_bridges': 'water_bridge',
        'salt_bridges': 'salt_bridge',
        'pi_stacks': 'pi_stack',
        'pi_cation_interactions': 'pi_cation_interaction',
        'halogen_bonds': 'halogen_bond',
        'metal_complexes': 'metal_complex',
}

def generate_plif_fp(report_file, lig_ident):
    '''inspired from https://www.blopig.com/blog/2016/11/how-to-calculate-plifs-using-rdkit-and-plip/'''
    try:
        tree = ET.parse(report_file)
    except ET.ParseError:
        print(f'Error parsing {report_file}')
        return None, None
    root = tree.getroot()

    #list of residue keys that form an interaction
    binding_site = root.findall('bindingsite')[0]
    plif_fp = []
    nest = binding_site.find('identifiers')
    lig_code = nest.find('hetid')
    coords = None

    if str(lig_code.text) == str(lig_ident):
        #get the plifs stuff here
        nest_residue = binding_site.find('bs_residues')
        residue_list_tree = nest_residue.findall('bs_residue')
        interactions = binding_site.find('interactions')

        hbonds = interactions.find('hydrogen_bonds').findall('hydrogen_bond')
        coords = torch.zeros(len(hbonds), 3)
        for i, hbond in enumerate(hbonds):
                coords[i, 0] = float(hbond.find('ligcoo').find('x').text)
                coords[i, 1] = float(hbond.find('ligcoo').find('y').text)
                coords[i, 2] = float(hbond.find('ligcoo').find('z').text)
        if coords.shape[0] == 0:
                coords = None

        contacts = []
        for residue in residue_list_tree:
                contact = residue.attrib['contact'] == 'True'
                contacts.append(contact)

        any_contact = any(contacts)

        # get binary if interaction present
        for key, value in interaction_types.items():
                interaction_list = interactions.find(key).findall(value)
                if interaction_list:
                        plif_fp.append(1)
                else:
                        plif_fp.append(0)

        # convert any_contact to binary
        if any_contact:
                plif_fp.append(1)
        else:
                plif_fp.append(0)

    return plif_fp, coords


def combine_mol_pdb(mol_pdb, protein_pdb, outfile):
    parser = PDBParser(QUIET=True)
    # Load the first PDB file
    structure1 = parser.get_structure('1', mol_pdb)
    # Load the second PDB file
    structure2 = parser.get_structure('2', protein_pdb)
    # Merge the structures
    for model in structure2:
        for chain in model:
            structure1[0].add(chain)
    # Create a PDBIO to write the merged structure to a file
    io = PDBIO()
    io.set_structure(structure1)
    io.save(outfile)

if __name__ == "__main__":
    args = parser.parse_args()

    # Read the SDF file
    sdf_supp = Chem.SDMolSupplier(args.sdf_file)

    # Read the CSV file
    df = pd.read_csv(args.csv_file)

    cols2keep = ['uuid', 'pdb', 'chain']
    df = df[cols2keep]

    all_plif_fps = []

    hbonds_dir = Path(args.output_dir) / 'hbonds'
    hbonds_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over the molecules
    for uuid, pdb, chain in zip(df['uuid'], df['pdb'], df['chain']):
        # outputdir
        output_dir = Path(args.output_dir) / f'{pdb}_{chain}_{uuid}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # check if report already exists
        plip_report = output_dir / 'report.xml'
        mol_pdb = output_dir / f'uuid_{uuid}.pdb'
        combined_pdb = Path(args.output_dir) / f'{pdb}_{chain}_{uuid}.pdb'

        if not plip_report.exists():
                # Get the molecule and cinvert to pdb
                mol = sdf_supp[int(uuid)]
                Chem.MolToPDBFile(mol, str(mol_pdb))

                # Get the combined molecule and protein pdb
                protein_path = Path(args.pdb_dir) / f'{pdb}_{chain}.pdb'
                combine_mol_pdb(mol_pdb, protein_path, str(combined_pdb))

                # Submit command to generate PLIP report
                command = f'plipcmd -f {combined_pdb} -o {output_dir} -x'
                subprocess.run(command, shell=True)

        # Generate the PLIF fingerprint
        plif_fp, coords = generate_plif_fp(plip_report, 'UNL')
        if plif_fp:
                all_plif_fps.append(plif_fp)
        else:
                all_plif_fps.append([999]*(len(interaction_types) + 1))

        if coords is not None:
                torch.save(coords, hbonds_dir / f'{pdb}_{chain}_{uuid}.pt')

        # clean up to free up storage
        if mol_pdb.exists():
                mol_pdb.unlink()
        if combined_pdb.exists():
                combined_pdb.unlink()
        # delete everything ending in .pdb in the output directory
        for file in output_dir.glob('*.pdb'):
            if file.exists():
                file.unlink()

    # Save the PLIF fingerprints as .pt file
    plif_fp = torch.tensor(all_plif_fps)
    torch.save(plif_fp, Path(args.output_dir) / 'plif_fp.pt')

    print('Done')
