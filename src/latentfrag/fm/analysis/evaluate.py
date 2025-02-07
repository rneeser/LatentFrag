import os
import sys
import re
from pathlib import Path
from typing import Collection, List, Dict, Type, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from latentfrag.fm.analysis.sbdd_metrics import FullEvaluator, FullCollectionEvaluator
from latentfrag.fm.analysis.frag_metrics import (SimpleFragmentEvaluator,
                                              ComplexFragmentEvaluator,
                                              FragNCIEvaluator)

AUXILIARY_COLUMNS = ['sample', 'sdf_file', 'pdb_file', 'subdir', 'x_h_pred_file', 'x_h_gt_file', 'idx']
VALIDITY_METRIC_NAME = 'medchem.valid'


def get_data_type(key: str, data_types: Dict[str, Type], default=None) -> Type:
    found_data_type_key = None
    found_data_type_value = None
    for data_type_key, data_type_value in data_types.items():
        if re.match(data_type_key, key) is not None:
            if found_data_type_key is not None:
                raise ValueError(f'Multiple data type keys match [{key}]: {found_data_type_key}, {data_type_key}')

            found_data_type_value = data_type_value
            found_data_type_key = data_type_key

    if found_data_type_key is None:
        if default is None:
            raise KeyError(key)
        else:
            found_data_type_value = default

    return found_data_type_value


def convert_data_to_table(data: List[Dict], data_types: Dict[str, Type]) -> pd.DataFrame:
    """
    Converts data to a detailed table
    """
    table = []
    for entry in data:
        table_entry = {}
        for key, value in entry.items():
            if key in AUXILIARY_COLUMNS:
                table_entry[key] = value
                continue
            if get_data_type(key, data_types) != list:
                table_entry[key] = value
        table.append(table_entry)

    return pd.DataFrame(table)


def aggregated_metrics(table: pd.DataFrame, data_types: Dict[str, Type], validity_metric_name: str = None):
    """
    Args:
        table (pd.DataFrame): table with metrics computed for each sample
        data_types (Dict[str, Type]): dictionary with data types for each column
        validity_metric_name (str): name of the column that has validity metric

    Returns:
        agg_table (pd.DataFrame): table with columns ['metric', 'value', 'std']
    """
    aggregated_results = []

    # If validity column name is provided:
    #    1. compute validity on the entire data
    #    2. drop all invalid molecules to compute the rest
    if validity_metric_name is not None:
        aggregated_results.append({
            'metric': validity_metric_name,
            'value': table[validity_metric_name].fillna(False).astype(float).mean(),
            'std': None,
        })
        table = table[table[validity_metric_name]]

    # Compute aggregated metrics + standard deviations where applicable
    for column in table.columns:
        if column in AUXILIARY_COLUMNS + [validity_metric_name] or get_data_type(column, data_types) == str:
            continue
        if get_data_type(column, data_types) == bool:
            values = table[column].fillna(0).values.astype(float).mean()
            std = None
        else:
            values = table[column].dropna().values.astype(float).mean()
            std = table[column].dropna().values.astype(float).std()

        aggregated_results.append({
            'metric': column,
            'value': values,
            'std': std,
        })

    agg_table = pd.DataFrame(aggregated_results)
    return agg_table


def collection_metrics(
        table: pd.DataFrame,
        reference_smiles: Collection[str],
        validity_metric_name: str = None,
        exclude_evaluators: Collection[str] = [],
):
    """
    Args:
        table (pd.DataFrame): table with metrics computed for each sample
        reference_smiles (Collection[str]): list of reference SMILES (e.g. training set)
        validity_metric_name (str): name of the column that has validity metric
        exclude_evaluators (Collection[str]): Evaluator IDs to exclude

    Returns:
        col_table (pd.DataFrame): table with columns ['metric', 'value']
    """

    # If validity column name is provided drop all invalid molecules
    if validity_metric_name is not None:
        table = table[table[validity_metric_name]]

    evaluator = FullCollectionEvaluator(reference_smiles, exclude_evaluators=exclude_evaluators)
    smiles = table['representation.smiles'].values
    if len(smiles) == 0:
        print('No valid input molecules')
        return pd.DataFrame(columns=['metric', 'value'])

    collection_metrics = evaluator(smiles)
    results = [
        {'metric': key, 'value': value}
        for key, value in collection_metrics.items()
    ]

    col_table = pd.DataFrame(results)
    return col_table


def evaluate_subdir(
        in_dir: Path,
        evaluator: FullEvaluator,
        desc: str = None,
        n_samples: int = None,
) -> List[Dict]:
    """
    Computes per-molecule metrics for a single directory of samples for one target
    """
    results = []
    valid_files = [
        int(fname.split('_')[0])
        for fname in os.listdir(in_dir)
        if fname.endswith('_ligand.sdf') and not fname.startswith('.')
    ]
    if len(valid_files) == 0:
        return pd.DataFrame()

    upper_bound = max(valid_files) + 1
    if n_samples is not None:
        upper_bound = min(upper_bound, n_samples)

    for i in tqdm(range(upper_bound), desc=desc, file=sys.stdout):
        in_mol = Path(in_dir, f'{i}_ligand.sdf')
        in_prot = Path(in_dir, f'{i}_pocket.pdb')
        res = evaluator(in_mol, in_prot)

        if isinstance(res, dict):
            res['sample'] = i
            res['sdf_file'] = str(in_mol)
            res['pdb_file'] = str(in_prot)
            results.append(res)
        elif isinstance(res, list):
            results.extend(res)

    return results, [None] * upper_bound


def evaluate_subdir_frags_simple(
        in_dir: Path,
        evaluator: SimpleFragmentEvaluator,
        desc: str = None,
        n_samples: int = None,
) -> List[Dict]:
    """
    Computes per-molecule metrics for a single directory of samples for one target
    """
    results = []
    valid_files = [
        int(fname.split('_')[0])
        for fname in os.listdir(in_dir)
        if fname.endswith('_x_h_pred.pt') and not fname.startswith('.')
    ]
    if len(valid_files) == 0:
        return pd.DataFrame()

    upper_bound = max(valid_files) + 1
    if n_samples is not None:
        upper_bound = min(upper_bound, n_samples)

    for i in tqdm(range(upper_bound), desc=desc, file=sys.stdout):
        pred_fn = Path(in_dir, f'{i}_x_h_pred.pt')
        gt_fn = Path(in_dir, f'{i}_x_h_gt.pt')
        pred = torch.load(pred_fn)
        gt = torch.load(gt_fn)
        smi_pred_fn = Path(in_dir, f'{i}_smiles_pred.txt')
        smi_gt_fn = Path(in_dir, f'{i}_smiles_gt.txt')
        pred_smiles = smi_pred_fn.read_text().strip()
        gt_smiles = smi_gt_fn.read_text().strip()
        pred_smiles = pred_smiles.split('.')
        gt_smiles = gt_smiles.split('.')
        res = evaluator(pred, gt, pred_smiles, gt_smiles)

        res['sample'] = i
        res['x_h_pred_file'] = str(pred_fn)
        res['x_h_gt_file'] = str(gt_fn)
        results.append(res)

    return results, [None] * upper_bound


def evaluate_subdir_frags_complex(
        in_dir: Path,
        evaluator: ComplexFragmentEvaluator,
        desc: str = None,
        n_samples: int = None,
) -> List[Dict]:
    """
    Computes per-molecule metrics for a single directory of samples for one target
    """
    results = []
    valid_files = [
        int(fname.split('_')[0])
        for fname in os.listdir(in_dir)
        if fname.endswith('_ligand.sdf') and not fname.startswith('.')
    ]
    if len(valid_files) == 0:
        return pd.DataFrame()

    upper_bound = max(valid_files) + 1
    if n_samples is not None:
        upper_bound = min(upper_bound, n_samples)

    rmsds = []
    cossims = []

    for i in tqdm(range(upper_bound), desc=desc, file=sys.stdout):
        in_mol = Path(in_dir, f'{i}_ligand.sdf')
        res, sim_corr = evaluator(in_mol)

        results.append(res)
        rmsds.extend(sim_corr[0])
        cossims.extend(sim_corr[1])

    return results, (torch.stack(rmsds), torch.stack(cossims))


def evaluate_all(
        in_dir: Path,
        evaluator: FullEvaluator,
        subdir_fn: Callable,
        n_samples: int = None,
        job_id: int = 0,
        n_jobs: int = 1,
) -> List[Dict]:
    """
    1. Computes per-molecule metrics for all single directories of samples
    2. Aggregates these metrics
    3. Computes additional collection metrics (if `reference_smiles_path` is provided)
    """
    extra_results = []
    data = []
    total_number_of_subdirs = len([path for path in in_dir.glob("[!.]*") if os.path.isdir(path)])
    i = 0
    all_dirs_sorted = sorted(
        [path for path in in_dir.glob("[!.]*") if os.path.isdir(path)]
    )

    for subdir in all_dirs_sorted:

        i += 1
        if (i - 1) % n_jobs != job_id:
            continue

        curr_data, extra_res = subdir_fn(
            in_dir=subdir,
            evaluator=evaluator,
            desc=f'[{i}/{total_number_of_subdirs}] {str(subdir.name)}',
            n_samples=n_samples,
        )
        for entry in curr_data:
            entry['subdir'] = str(subdir)
            data.append(entry)

        if extra_res is not None:
            extra_results.append(extra_res)

    return data, extra_results


def compute_all_metrics(
        in_dir: Path,
        set_name: str,
        use_ev: bool,
        gnina_path: Path,
        plip_exec: Path,
        reference_path: Path = None,
        n_samples: int = None,
        validity_metric_name: str = VALIDITY_METRIC_NAME,
        exclude_evaluators: Collection[str] = [],
        job_id: int = 0,
        n_jobs: int = 1,
        trained_dmasif: str = None,
        frag_encoder_params: Dict = None,
):
    reference_smiles_path = Path(reference_path, 'train_smiles.npy')
    reference_structure_path = Path(reference_path, set_name)

    evaluator = FullEvaluator(gnina=gnina_path, plip_exec=plip_exec,exclude_evaluators=exclude_evaluators)
    data, _ = evaluate_all(in_dir=in_dir, evaluator=evaluator, subdir_fn=evaluate_subdir,
                        n_samples=n_samples, job_id=job_id, n_jobs=n_jobs)

    all_dtypes = evaluator.dtypes
    table_detailed = convert_data_to_table(data, all_dtypes)

    frag_evaluator = SimpleFragmentEvaluator()
    frag_results, _ = evaluate_all(in_dir=in_dir, evaluator=frag_evaluator,
                                             subdir_fn=evaluate_subdir_frags_simple,n_samples=n_samples,
                                             job_id=job_id, n_jobs=n_jobs)
    frag_results = convert_data_to_table(frag_results, frag_evaluator.dtypes)
    frag_results = frag_results.drop(columns=['sample', 'subdir'])

    if trained_dmasif is not None:
        frag_eval_complex = ComplexFragmentEvaluator(trained_dmasif, use_ev=use_ev,
                                                     frag_encoder_params=frag_encoder_params,
                                                     ref_dir=reference_structure_path)
        frag_results_complex, rmsds_cossims = evaluate_all(in_dir=in_dir, evaluator=frag_eval_complex,
                                                subdir_fn=evaluate_subdir_frags_complex,n_samples=n_samples,
                                                job_id=job_id, n_jobs=n_jobs)
        frag_results_complex = convert_data_to_table(frag_results_complex, frag_eval_complex.dtypes)
        frag_results_complex = frag_results_complex.drop(columns=['subdir'])
        rmsds = np.concatenate([r[0].numpy() for r in rmsds_cossims])
        cossims = np.concatenate([r[1].numpy() for r in rmsds_cossims])

        corr = np.corrcoef(rmsds, cossims)[0][1]
        corr_df = pd.DataFrame([{'metric': 'frag_metrics_complex.rmsd_cossim_corr', 'value': corr}])
        # save rmsds and cossims
        np.save(Path(in_dir, 'rmsds.npy'), rmsds)
        np.save(Path(in_dir, 'cossims.npy'), cossims)

    else:
        frag_results_complex = pd.DataFrame()
        frag_eval_complex.dtypes = {}
        rmsds = None
        cossims = None

    table_detailed_all = pd.concat([table_detailed, frag_results, frag_results_complex], axis=1)
    all_dtypes = {**all_dtypes, **frag_evaluator.dtypes, **frag_eval_complex.dtypes}

    table_aggregated = aggregated_metrics(
        table_detailed_all,
        data_types=all_dtypes,
        validity_metric_name=validity_metric_name
    )

    frags_nci_evaluator_partial = FragNCIEvaluator(gnina=gnina_path,
                                                   plip_exec=plip_exec,
                                                   reference_ligand_dir=reference_structure_path,
                                                   state='dock_partial')
    frag_nci_results_partial, _ = evaluate_all(in_dir=in_dir,
                                               evaluator=frags_nci_evaluator_partial,
                                               subdir_fn=evaluate_subdir, n_samples=n_samples,
                                               job_id=job_id, n_jobs=n_jobs)

    table_detailed_frag_nci = convert_data_to_table(frag_nci_results_partial, frags_nci_evaluator_partial.dtypes)

    table_aggregated_frag_nci = aggregated_metrics(
        table_detailed_frag_nci,
        data_types=frags_nci_evaluator_partial.dtypes,
    )

    table_aggregated = pd.concat([table_aggregated, table_aggregated_frag_nci])

    # Add collection metrics (uniqueness, novelty, FCD, etc.) if reference smiles are provided
    if reference_smiles_path is not None:
        reference_smiles = np.load(reference_smiles_path)
        col_metrics = collection_metrics(
            table=table_detailed_all,
            reference_smiles=reference_smiles,
            validity_metric_name=validity_metric_name,
            exclude_evaluators=exclude_evaluators
        )
        table_aggregated = pd.concat([table_aggregated, col_metrics])

    if trained_dmasif is not None:
        table_aggregated = pd.concat([table_aggregated, corr_df])

    return data, table_detailed_all, table_aggregated, table_detailed_frag_nci
