import sys
import json
import os
import requests
from pathlib import Path

from tqdm import tqdm


URL_entity = "https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}"
URL_PDB = "https://data.rcsb.org/rest/v1/core/entry/{}"


def get_mapping_for_pdb(pdb_code):
    mapping = {pdb_code: {}}
    response_pdb = requests.get(URL_PDB.format(pdb_code))
    if response_pdb.status_code != 200:
        return mapping

    polymer_entity_ids = response_pdb.json()['rcsb_entry_container_identifiers']['polymer_entity_ids']
    for numeric_entity_id in polymer_entity_ids:
        response_entity = requests.get(URL_entity.format(pdb_code, numeric_entity_id))
        if response_entity.status_code == 200:
            auth_chain_ids = response_entity.json()['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
            for auth_chain_id in auth_chain_ids:
                if auth_chain_id not in mapping[pdb_code]:
                    mapping[pdb_code][auth_chain_id] = numeric_entity_id

    return mapping


if __name__ == '__main__':

    pdb_codes = list(set([x.stem.split('_')[0] for x in Path(sys.argv[1]).glob('*.pdb') if x.stem.count('_') == 1]))
    print(f'Input directory contains {len(pdb_codes)} files')

    output_json_path = sys.argv[2]
    if os.path.exists(output_json_path):
        with open(output_json_path) as f:
            all_mapping = json.load(f)
        pdb_codes = [pdb for pdb in pdb_codes if pdb not in all_mapping.keys()]
        print(f'Found existing output file containing {len(all_mapping)} pdbs.')
        print(f'Going to process remaining {len(pdb_codes)} files')
    else:
        all_mapping = {}

    for i, pdb_code in enumerate(tqdm(pdb_codes)):
        all_mapping.update(get_mapping_for_pdb(pdb_code))
        if i % 1000 == 0:
            with open(output_json_path, "w") as outfile:
                json.dump(all_mapping, outfile)

    print("Writing out the final results")
    with open(sys.argv[2], "w") as outfile:
        json.dump(all_mapping, outfile)

    print("Done")
