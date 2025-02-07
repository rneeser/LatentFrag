import argparse
from pathlib import Path

import xml.etree.ElementTree as ET
import torch
import torch.nn.functional as F

from latentfrag.encoder.utils.constants import NCIS


def get_interaction_info(interactions, interaction_types):
    res_ids = []
    prot_coords = []
    lig_coords = []
    dists = []
    nci_ohe = []
    for key, value in interaction_types.items():
        nci = interactions.find(key).findall(value)
        if nci is not None:
            for interaction in nci:
                prot_coord = torch.zeros(3)
                lig_coord = torch.zeros(3)
                res_ids.append(int(interaction.find('resnr').text))
                nci_ohe.append(F.one_hot(torch.tensor(list(interaction_types.values()).index(value)), len(interaction_types)))
                prot_coord[0] = float(interaction.find('protcoo').find('x').text)
                prot_coord[1] = float(interaction.find('protcoo').find('y').text)
                prot_coord[2] = float(interaction.find('protcoo').find('z').text)
                prot_coords.append(prot_coord)
                lig_coord[0] = float(interaction.find('ligcoo').find('x').text)
                lig_coord[1] = float(interaction.find('ligcoo').find('y').text)
                lig_coord[2] = float(interaction.find('ligcoo').find('z').text)
                lig_coords.append(lig_coord)
                if key == 'hydrogen_bonds':
                    dists.append(float(interaction.find('dist_d-a').text))
                elif key == 'pi_stacks':
                    dists.append(float(interaction.find('centdist').text))
                else:
                    dists.append(float(interaction.find('dist').text))
    if res_ids:
        res_ids = torch.tensor(res_ids)
        prot_coords = torch.stack(prot_coords)
        lig_coords = torch.stack(lig_coords)
        dists = torch.tensor(dists)
        nci_ohe = torch.stack(nci_ohe)
        return res_ids, prot_coords, lig_coords, dists, nci_ohe
    else:
        return None, None, None, None, None


def get_interactions(fn):
    try:
        tree = ET.parse(fn)
    except ET.ParseError:
        print(f'Error parsing {fn}')
    root = tree.getroot()
    binding_site = root.findall('bindingsite')[0]
    interactions = binding_site.find('interactions')
    return interactions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plip_dir', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    assert args.plip_dir.exists(), f'{args.plip_dir} does not exist'

    args.out.mkdir(exist_ok=True, parents=True)

    chain_ncis = {}

    for report_file in args.plip_dir.glob('*/*.xml'):
        interactions = get_interactions(report_file)

        res_ids, prot_coords, lig_coords, dists, nci_ohe = get_interaction_info(interactions, NCIS)

        pdb_id, chain_id, uuid = str(report_file).split('/')[-2].split('_')
        if f'{pdb_id}_{chain_id}' not in chain_ncis:
            chain_ncis[f'{pdb_id}_{chain_id}'] = {}
        if int(uuid) not in chain_ncis[f'{pdb_id}_{chain_id}']:
            chain_ncis[f'{pdb_id}_{chain_id}'][int(uuid)] = {
                'res_ids': res_ids,
                'prot_coords': prot_coords,
                'lig_coords': lig_coords,
                'dists': dists,
                'nci_ohe': nci_ohe
            }

    for chain, uuids in chain_ncis.items():
        for uuid, nci_info in uuids.items():
            with open(args.out / f'{chain}_{uuid}.pt', 'wb') as f:
                torch.save(nci_info, f)

    print('Done')