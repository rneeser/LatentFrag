import argparse
import json

import pandas as pd

def get_interaction_entity(row, entities):
    pdb = row['pdb']
    chain = row['chain']
    entity = entities.get(pdb.upper(), {}).get(chain)
    return entity


def get_interaction_cluster(row, clusters):
    pdb = row['pdb'].upper()
    entity = row['entity']
    cluster = clusters.get(pdb.upper(), {}).get(entity)
    return cluster


def split(
        pdb_interactions_path,
        pdb_entities_path,
        holoprot_splits_path,
        clusters_path,
        output_dir,
):
    pdb_interactions = pd.read_csv(pdb_interactions_path)

    pdb_entities = json.load(open(pdb_entities_path))
    holoprot_splits = json.load(open(holoprot_splits_path))

    clusters = {}
    with open(clusters_path) as f:
        for clust, line in enumerate(f.readlines()):
            for item in line.strip().split():
                if item.count('_') == 1:
                    pdbid, entity = item.split('_')
                    clusters.setdefault(pdbid, {})
                    clusters[pdbid][entity] = clust

    pdb_interactions['entity'] = pdb_interactions.apply(lambda row: get_interaction_entity(row, pdb_entities), axis=1)
    pdb_interactions['cluster'] = pdb_interactions.apply(lambda row: get_interaction_cluster(row, clusters), axis=1)
    print('Interactions:', len(pdb_interactions))
    pdb_interactions = pdb_interactions[pdb_interactions.cluster.notna()]
    print('Interactions no NA:', len(pdb_interactions))

    # make elems in value lists upper in holoprot_splits
    holoprot_splits = {k: [x.upper() for x in v] for k, v in holoprot_splits.items()}

    holoprot_cluster_splits = {
        s: pdb_interactions[pdb_interactions.pdb.isin(pdbs)].cluster.dropna().astype(int).unique().tolist()
        for s, pdbs in holoprot_splits.items()
    }
    pdb_interactions['split'] = pdb_interactions.cluster.apply(
        lambda c: (
            None if pd.isna(c) else
            'test' if int(c) in holoprot_cluster_splits['test'] else
            'valid' if int(c) in holoprot_cluster_splits['valid'] else
            'train'
        )
    )

    train = {
        'pdb': [],
        'chain': [],
        'cluster': [],
        'idx' : []
    }

    test = {
        'pdb': [],
        'chain': [],
        'cluster': [],
        'idx' : []
    }

    val = {
        'pdb': [],
        'chain': [],
        'cluster': [],
        'idx' : []
    }

    for row in pdb_interactions.itertuples():
        lig_idx = row.idx
        pdb_id = row.pdb
        chain = row.chain
        cluster = row.cluster
        split = row.split

        if split == 'train':

            train['pdb'].append(pdb_id)
            train['chain'].append(chain)
            train['cluster'].append(cluster)
            train['idx'].append(lig_idx)

        elif split == 'test':

            test['pdb'].append(pdb_id)
            test['chain'].append(chain)
            test['cluster'].append(cluster)
            test['idx'].append(lig_idx)

        elif split == 'valid':

            val['pdb'].append(pdb_id)
            val['chain'].append(chain)
            val['cluster'].append(cluster)
            val['idx'].append(lig_idx)

    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    val_df = pd.DataFrame(val)

    train_df = train_df.drop_duplicates(keep='first')
    test_df = test_df.drop_duplicates(keep='first')
    val_df = val_df.drop_duplicates(keep='first')

    train_df.to_csv(f'{output_dir}/train_fbdd_ids.csv', index=False)
    test_df.to_csv(f'{output_dir}/test_fbdd_ids.csv', index=False)
    val_df.to_csv(f'{output_dir}/val_fbdd_ids.csv', index=False)

    print('Train:', len(train_df))
    print('Test:', len(test_df))
    print('Val:', len(val_df))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--pdb_interactions', type=str, required=True)
    p.add_argument('--pdb_entities', type=str, required=True)
    p.add_argument('--holoprot_splits', type=str, required=True)
    p.add_argument('--clusters', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    args = p.parse_args()

    split(
        args.pdb_interactions,
        args.pdb_entities,
        args.holoprot_splits,
        args.clusters,
        args.output_dir,
    )
