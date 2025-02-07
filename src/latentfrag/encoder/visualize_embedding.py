from pathlib import Path
import argparse

import pandas as pd
from rdkit import Chem
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import rankdata, spearmanr
import numpy as np

from latentfrag.encoder.models.lightning_modules import FragEmbed
from latentfrag.encoder.utils.data import featurize_ligand, TensorDict
from latentfrag.encoder.utils.io import pointcloud_to_pdb


def get_embedding(mol, model, use_ev):
    frag_feat = featurize_ligand(mol, use_ev)

    frag_feat = TensorDict(**frag_feat)
    frag_feat['batch'] = torch.zeros(len(frag_feat['xyz']), dtype=int)
    frag_feat = frag_feat.to(device)
    frag_feat['desc'], frag_feat['desc_global'] = model.ligand_encoder(
            frag_feat["xyz"], frag_feat["types"], frag_feat["batch"],
            frag_feat["bonds"], frag_feat["bond_types"], frag_feat["mol_feats"], return_global=True)
    return frag_feat

def top_k_accuracy_with_cutoff(similarities, distances, k, distance_cutoff):
    # Compute cosine similarities between mol and all surface points
    top_k_indices = torch.argsort(similarities)[-k:]

    # Check how many top k points are within distance cutoff
    correct = torch.sum(distances[top_k_indices] <= distance_cutoff)
    return (correct / k).item()


def compute_aupr(similarities, distances, distance_cutoff):
    true_labels = (distances <= distance_cutoff).int()
    if torch.sum(true_labels) == 0:
        return np.nan
    precision, recall, _ = precision_recall_curve(true_labels.cpu().numpy(), similarities.cpu().numpy())
    return auc(recall, precision)


def enrichment_factor(similarities, distances,
                     similarity_percentile, distance_cutoff):
    sim_threshold = torch.quantile(similarities, similarity_percentile)

    # Count points above similarity threshold
    high_sim_points = similarities >= sim_threshold
    n_high_sim = torch.sum(high_sim_points)

    # Count binding sites among high similarity points
    n_high_sim_binding = torch.sum((distances <= distance_cutoff) & high_sim_points)

    # Count total binding sites
    n_total_binding = torch.sum(distances <= distance_cutoff)
    total_points = len(distances)

    # Calculate enrichment factor:
    # (binding sites in high sim / total high sim) / (total binding sites / total points)
    if n_high_sim == 0:
        return 0.0
    elif n_total_binding == 0:
        return np.nan

    observed_fraction = n_high_sim_binding / n_high_sim
    expected_fraction = n_total_binding / total_points

    return (observed_fraction / expected_fraction).item()


def mean_reciprocal_rank(similarities, distances, distance_cutoff):
    ranks = rankdata(-similarities.cpu().numpy()) # Higher similarity = lower rank
    binding_ranks = ranks[distances.cpu().numpy() <= distance_cutoff]
    if len(binding_ranks) == 0:
        return np.nan
    return np.mean(1 / binding_ranks)


def distance_weighted_accuracy(similarities, distances,
                             max_distance, temperature=1.0):
    # Convert distances to weights (closer points matter more)
    weights = torch.exp(-(distances / max_distance) / temperature)
    weights = weights / torch.sum(weights)  # normalize

    # Compute correlation between similarities and weights
    return spearmanr(similarities.cpu(), weights.cpu()).correlation


def calc_metrics(similarities, min_distances, distance_cutoff):
    top_1_acc = top_k_accuracy_with_cutoff(similarities, min_distances, 1, distance_cutoff)
    top_10_acc = top_k_accuracy_with_cutoff(similarities, min_distances, 10, distance_cutoff)
    top_100_acc = top_k_accuracy_with_cutoff(similarities, min_distances, 100, distance_cutoff)
    aupr = compute_aupr(similarities, min_distances, distance_cutoff)
    ef_1 = enrichment_factor(similarities, min_distances, 0.99, distance_cutoff)
    ef_5 = enrichment_factor(similarities, min_distances, 0.95, distance_cutoff)
    ef_10 = enrichment_factor(similarities, min_distances, 0.9, distance_cutoff)
    mrr = mean_reciprocal_rank(similarities, min_distances, distance_cutoff)
    dwa = distance_weighted_accuracy(similarities, min_distances, 9.0)
    return top_1_acc, top_10_acc, top_100_acc, aupr, ef_1, ef_5, ef_10, mrr, dwa


parser = argparse.ArgumentParser()

parser.add_argument('--ckpt', type=Path, required=True, help='Path to the model checkpoint')
parser.add_argument('--out', type=Path, required=True, help='Path to the output directory')
parser.add_argument('--pdb_dir', type=Path, required=True, help='Path to the directory containing PDB files')
parser.add_argument('--full_ligand_dir', type=Path, required=False, help='Path to the directory containing full ligands', default=None)
parser.add_argument('--fragments', type=str, required=True, help='Path to the sdf file containing fragments')
parser.add_argument('--csvpath', type=Path, required=True, help='Path to the csv file containing the ligand to chain mapping')
parser.add_argument('--samples', type=int, default=100, help='Number of chains to visualize')
parser.add_argument('--msms_bin', type=Path, required=False, help='Path to the MSMS binary')
parser.add_argument('--msms_res', type=float, default=1.0, help='MSMS resolution')
parser.add_argument('--msms_subsampling', type=int, default=20, help='MSMS sup-sampling')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

# make output directory
args.out.mkdir(parents=True, exist_ok=True)

pdb_dir = args.pdb_dir / 'chains_protonated'

assert args.ckpt.exists(), 'Checkpoint file not found'
assert pdb_dir.exists(), 'PDB directory not found'
assert args.csvpath.exists(), 'CSV file not found'
assert Path(args.fragments).exists(), 'Fragments file not found'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = FragEmbed.load_from_checkpoint(args.ckpt,
                                    map_location=device,
                                    dataset_params={'pdb_dir': pdb_dir,
                                                    'preprocess': False,
                                                    'datadir': ''},
                                    surface_params={'msms_bin': args.msms_bin,
                                                    'resolution': args.msms_res,
                                                    'sup_sampling': args.msms_subsampling},
                                    strict=False
                                    )
model.to(device)
model.eval()
# get use_ev from hparams
use_ev = model.hparams.get('ligand_encoder_params').get('use_ev', False)
print(f'Using exit vectors: {use_ev}')

df = pd.read_csv(args.csvpath)

# group by pdb and chain
pdb_chain = df.groupby(['pdb', 'chain'])
pdb_chain_uuid = pdb_chain['uuid'].apply(list)
pdb_chain_idx = pdb_chain['idx'].apply(list)


# sample chains
chains = pdb_chain_uuid.sample(n=args.samples, random_state=args.seed)

mol_suppl = Chem.SDMolSupplier(args.fragments)

metrics = {
    'pdb': [],
    'uuid': [],
    'n_all': [],
    'n_true': [],
    'true_frac': [],
    'top_1_acc': [],
    'top_10_acc': [],
    'top_100_acc': [],
    'aupr': [],
    'ef_1': [],
    'ef_5': [],
    'ef_10': [],
    'mrr': [],
    'dwa': [],
    # pocket metrics
    'n_all_pocket': [],
    'n_true_pocket': [],
    'true_frac_pocket': [],
    'top_1_acc_pocket': [],
    'top_10_acc_pocket': [],
    'top_100_acc_pocket': [],
    'aupr_pocket': [],
    'ef_1_pocket': [],
    'ef_5_pocket': [],
    'ef_10_pocket': [],
    'mrr_pocket': [],
    'dwa_pocket': [],
}

all_pocket_sims = []
all_pocket_dists = []

# iterate over chains
for (pdb, chain), uuids in tqdm(chains.iteritems(), total=args.samples):
    pdb_path = pdb_dir / f'{pdb}_{chain}.pdb'
    indices = pdb_chain_idx[(pdb, chain)]

    try:
        protein = model.encode_protein(pdb_path)

        if args.full_ligand_dir:
            ligand_path = args.full_ligand_dir / f'{pdb}.sdf'
            ligand = Chem.SDMolSupplier(str(ligand_path), sanitize=True)
            masks = {}
            for n in set(indices):
                lig = ligand[n]
                coords = torch.tensor(lig.GetConformer().GetPositions(), device=device, dtype=torch.float32)
                lig_dists = torch.cdist(coords, protein['xyz'])
                mask = lig_dists.min(0).values < 7.0
                masks[n] = mask

        for uuid, idx in zip(uuids, indices):
            frag = mol_suppl[int(uuid)]

            frag_coords = torch.tensor(frag.GetConformer().GetPositions(), device=device, dtype=torch.float32)
            dists = torch.cdist(protein['xyz'], frag_coords.unsqueeze(0)).squeeze()
            # min distances per surface point
            min_distances = torch.min(dists, dim=1).values

            frag_feat = get_embedding(frag, model, use_ev)

            cosine_sims = F.cosine_similarity(protein['desc'], frag_feat['desc_global'].repeat(len(protein['desc']), 1), dim=1)
            cosine_sims_scaled = (cosine_sims + 1) / 2

            top_1_acc, top_10_acc, top_100_acc, aupr, ef_1, ef_5, ef_10, mrr, dwa = \
                calc_metrics(cosine_sims.detach(), min_distances, 3.0)

            n_all = len(cosine_sims)
            n_true = torch.sum(min_distances <= 3.0).item()

            metrics['pdb'].append(protein['name'].split('.')[0])
            metrics['uuid'].append(uuid)
            metrics['n_all'].append(n_all)
            metrics['n_true'].append(n_true)
            metrics['true_frac'].append(n_true / n_all)
            metrics['top_1_acc'].append(top_1_acc)
            metrics['top_10_acc'].append(top_10_acc)
            metrics['top_100_acc'].append(top_100_acc)
            metrics['aupr'].append(aupr)
            metrics['ef_1'].append(ef_1)
            metrics['ef_5'].append(ef_5)
            metrics['ef_10'].append(ef_10)
            metrics['mrr'].append(mrr)
            metrics['dwa'].append(dwa)

            if args.full_ligand_dir:
                mask = masks[idx]
                cosine_sims = cosine_sims[mask]
                min_distances = min_distances[mask]
                top_1_acc_pocket, top_10_acc_pocket, top_100_acc_pocket, \
                    aupr_pocket, ef_1_pocket, ef_5_pocket, ef_10_pocket, mrr_pocket, dwa_pocket = \
                        calc_metrics(cosine_sims.detach(), min_distances, 3.0)

                all_pocket_sims.append(cosine_sims)
                all_pocket_dists.append(min_distances)

                n_all_pocket = len(cosine_sims)
                n_true_pocket = torch.sum(min_distances <= 3.0).item()

                metrics['n_all_pocket'].append(n_all_pocket)
                metrics['n_true_pocket'].append(n_true_pocket)
                metrics['true_frac_pocket'].append(n_true_pocket / n_all_pocket)
                metrics['top_1_acc_pocket'].append(top_1_acc_pocket)
                metrics['top_10_acc_pocket'].append(top_10_acc_pocket)
                metrics['top_100_acc_pocket'].append(top_100_acc_pocket)
                metrics['aupr_pocket'].append(aupr_pocket)
                metrics['ef_1_pocket'].append(ef_1_pocket)
                metrics['ef_5_pocket'].append(ef_5_pocket)
                metrics['ef_10_pocket'].append(ef_10_pocket)
                metrics['mrr_pocket'].append(mrr_pocket)
                metrics['dwa_pocket'].append(dwa_pocket)

            outfile = args.out / f'{pdb}_{chain}_{uuid}.pdb'

            pointcloud_to_pdb(outfile, protein['xyz'], cosine_sims_scaled.detach().cpu().numpy())

            # frag to sdf
            frag_path = args.out / f'{pdb}_{chain}_{uuid}.sdf'
            with Chem.SDWriter(str(frag_path)) as writer:
                writer.write(frag)

    except Exception as e:
        print(f'Error processing {pdb}_{chain}: {e}')
        continue

if args.full_ligand_dir:
    all_pocket_dists = torch.cat(all_pocket_dists)
    all_pocket_sims = torch.cat(all_pocket_sims)
    torch.save(all_pocket_dists, args.out / 'all_pocket_dists.pt')
    torch.save(all_pocket_sims, args.out / 'all_pocket_sims.pt')

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(args.out / 'metrics.csv', index=False)
print(f'Metrics saved to {args.out / "metrics.csv"}')

print(f'Averaged metrics:')
print(metrics_df.mean())
