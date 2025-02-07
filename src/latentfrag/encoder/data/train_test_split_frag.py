import argparse
import pandas as pd
import json


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
        output_train_path,
        output_valid_path,
        output_test_path,
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

    train_clusters = set(pdb_interactions[pdb_interactions.split == 'train'].cluster.unique())
    valid_clusters = set(pdb_interactions[pdb_interactions.split == 'valid'].cluster.unique())
    test_clusters = set(pdb_interactions[pdb_interactions.split == 'test'].cluster.unique())

    all_pdbs = set(pdb_interactions.pdb.unique())
    train_pdbs = set(pdb_interactions[pdb_interactions.split == 'train'].pdb.unique())
    valid_pdbs = set(pdb_interactions[pdb_interactions.split == 'valid'].pdb.unique())
    test_pdbs = set(pdb_interactions[pdb_interactions.split == 'test'].pdb.unique())

    all_frags = set(pdb_interactions.frag_smi.unique())
    train_frags = set(pdb_interactions[pdb_interactions.split == 'train'].frag_smi.unique())
    valid_frags = set(pdb_interactions[pdb_interactions.split == 'valid'].frag_smi.unique())
    test_frags = set(pdb_interactions[pdb_interactions.split == 'test'].frag_smi.unique())

    print('Data points:')
    print('Train:', len(pdb_interactions[pdb_interactions.split == 'train']))
    print('Valid:', len(pdb_interactions[pdb_interactions.split == 'valid']))
    print('Test:', len(pdb_interactions[pdb_interactions.split == 'test']))
    print()

    print('Unique clusters:')
    print('Train:', len(train_clusters))
    print('Valid:', len(valid_clusters))
    print('Test:', len(test_clusters))
    print()

    print('Unique PDBs:')
    print('All:', len(all_pdbs))
    print('Train:', len(train_pdbs))
    print('Valid:', len(valid_pdbs))
    print('Test:', len(test_pdbs))
    print()

    print('Unique fragments:')
    print('All:', len(all_frags))
    print('Train:', len(train_frags))
    print('Valid:', len(valid_frags))
    print('Test:', len(test_frags))
    print()

    print('Clusters overlap:')
    print('Train/test:', len(train_clusters & test_clusters))
    print('Train/valid:', len(train_clusters & valid_clusters))
    print('Test/valid:', len(test_clusters & valid_clusters))
    print()

    print('PDBs overlap:')
    print('Train/test:', len(train_pdbs & test_pdbs))
    print('Train/valid:', len(train_pdbs & valid_pdbs))
    print('Test/valid:', len(test_pdbs & valid_pdbs))
    print()

    print('Fragments overlap:')
    print('Train/test:', len(train_frags & test_frags))
    print('Train/valid:', len(train_frags & valid_frags))
    print('Test/valid:', len(test_frags & valid_frags))
    print()

    pdb_interactions[pdb_interactions.split == 'train'].to_csv(output_train_path, index=False)
    pdb_interactions[pdb_interactions.split == 'valid'].to_csv(output_valid_path, index=False)
    pdb_interactions[pdb_interactions.split == 'test'].to_csv(output_test_path, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--pdb_interactions', type=str, required=True)
    p.add_argument('--pdb_entities', type=str, required=True)
    p.add_argument('--holoprot_splits', type=str, required=True)
    p.add_argument('--clusters', type=str, required=True)
    p.add_argument('--output_train', type=str, required=True)
    p.add_argument('--output_valid', type=str, required=True)
    p.add_argument('--output_test', type=str, required=True)
    args = p.parse_args()

    split(
        args.pdb_interactions,
        args.pdb_entities,
        args.holoprot_splits,
        args.clusters,
        args.output_train,
        args.output_valid,
        args.output_test,
    )