import argparse
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
import xml.etree.ElementTree as ET

ALLOWED_LIGAND_ATOMS = [
    'P', 'O', 'S', 'N', 'C', 'Cl', 'I', 'B', 'Br', 'F',
]
UNWANTED_SMARTS = ['[#6;!R]-[#6;!R]-[#6;!R]-[#6;!R]',
                   '[#8,#8&H1]=,-[#15](=,-[#8,#8&H1])(=,-[#8,#8&H1])=,-[#8,#8&H1]',
                   ]
UNWANTED_PATTERNS = [Chem.MolFromSmarts(smarts) for smarts in UNWANTED_SMARTS]
MAX_MW = 500
MAX_NUM_ATOMS = 20
MAX_RING_SIZE = 8


def filterbyatoms(mol, allowed_atoms):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True


def max_ringsize(mol):
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return max(ring_sizes) if ring_sizes else 0


def filterbypatterns(mol, unwanted_patterns):
    for pattern in unwanted_patterns:
        if mol.HasSubstructMatch(pattern):
            return False
    return True


def filterbyinteraction(row, interaction_dir):
    uuid = row['uuid']
    pdbid = row['pdb']
    chain = row['chain']
    report_path = Path(interaction_dir, f"{pdbid}_{chain}_{uuid}", 'report.xml')
    if not report_path.exists():
        return False
    try:
        tree = ET.parse(report_path)
        root = tree.getroot()

        #list of residue keys that form an interaction
        binding_site = root.findall('bindingsite')[0]

        nest_residue = binding_site.find('bs_residues')
        residue_list_tree = nest_residue.findall('bs_residue')

        contacts = []
        for residue in residue_list_tree:
            contact = residue.attrib['contact'] == 'True'
            contacts.append(contact)

        return any(contacts)
    except:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--csvpath', type=str, required=True)
    p.add_argument('--interaction_dir', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csvpath)

    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='frag_smi', molCol='mol', includeFingerprints=False)

    # filter by element types
    df['valid'] = df['mol'].apply(lambda x: filterbyatoms(x, ALLOWED_LIGAND_ATOMS))
    df_filtered = df[df['valid'] == True]

    # filter by properties
    df_filtered['num_atoms'] = df_filtered['mol'].apply(lambda x: x.GetNumAtoms())
    df_filtered['num_bonds'] = df_filtered['mol'].apply(lambda x: x.GetNumBonds())
    df_filtered['MW'] = df_filtered['mol'].apply(lambda x: Chem.Descriptors.MolWt(x))
    df_filtered['max_ring_size'] = df_filtered['mol'].apply(lambda x: max_ringsize(x))
    df_filtered = df_filtered[df_filtered['num_atoms'] <= MAX_NUM_ATOMS]
    df_filtered = df_filtered[df_filtered['MW'] <= MAX_MW]
    df_filtered = df_filtered[df_filtered['max_ring_size'] <= MAX_RING_SIZE]

    # filter by patterns (phosphate, alkyl chains)
    df_filtered['pattern_filter'] = df_filtered['mol'].apply(lambda x: filterbypatterns(x, UNWANTED_PATTERNS))
    df_filtered = df_filtered[df_filtered['pattern_filter'] == True]

    # filter by interaction
    df_filtered['interaction'] = df_filtered.apply(lambda x: filterbyinteraction(x, args.interaction_dir), axis=1)
    df_filtered_interact = df_filtered[df_filtered['interaction'] == True]

    # drop the mol column
    df_filtered = df_filtered.drop(columns=['mol'])
    df_filtered_interact = df_filtered_interact.drop(columns=['mol'])

    # save the filtered data
    df_filtered.to_csv(args.out.replace('.csv', '_all.csv'), index=False)
    df_filtered_interact.to_csv(args.out, index=False)
