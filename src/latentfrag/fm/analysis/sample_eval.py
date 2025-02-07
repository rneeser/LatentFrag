import argparse
import yaml
import pickle
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem
from tqdm import tqdm

from latentfrag.fm.utils.gen_utils import (dict_to_namespace,
                                        namespace_to_dict,
                                        set_deterministic,
                                        disable_rdkit_logging,
                                        write_sdf_file)
from latentfrag.fm.analysis.visualization_utils import mols_to_pdbfile
from latentfrag.fm.data.data_utils import TensorDict
from latentfrag.fm.models.lightning import LatentFrag
from latentfrag.fm.analysis.evaluate import compute_all_metrics


def combine(base_args, override_args):
    assert not isinstance(base_args, dict)
    assert not isinstance(override_args, dict)

    arg_dict = base_args.__dict__
    for key, value in override_args.__dict__.items():
        if key not in arg_dict or arg_dict[key] is None:  # parameter not provided previously
            arg_dict[key] = value
        elif isinstance(value, Namespace):
            arg_dict[key] = combine(arg_dict[key], value)
        else:
            arg_dict[key] = value
    return base_args


def path_to_str(input_dict):
    for key, value in input_dict.items():
        if isinstance(value, dict):
            input_dict[key] = path_to_str(value)
        else:
            input_dict[key] = str(value) if isinstance(value, Path) else value
    return input_dict


def get_closest_smiles(h_pred, library, lib_frags):
    """
    Get the closest smiles to the predicted fragment from the library based on cosine similarity.

    :param h_pred: torch tensor of shape (n, h_dim)
    :param library: dict of torch tensors (1, h_dim) to indices (int)
    :param lib_frags: list of RDKit mol objects (sorted by index)
    :return: RDKit mol object
    """
    lib_embeddings = torch.stack(list(library.keys()))

    similarities = F.cosine_similarity(h_pred.unsqueeze(0), lib_embeddings, dim=2)
    closest_idx = similarities.argmax(dim=0)

    uuids = list(library.values())
    closest_uuid = [uuids[idx] for idx in closest_idx]

    mols = [lib_frags[uuid] for uuid in closest_uuid]
    smiles = [Chem.MolToSmiles(m) for m in mols]
    return smiles


def sample(cfg, model_params, samples_dir, job_id=0, n_jobs=1):
    print('Sampling...')
    model = LatentFrag.load_from_checkpoint(cfg.checkpoint, map_location=cfg.device, **model_params)
    model.setup(stage='fit' if cfg.set == 'train' else cfg.set)
    model.eval().to(cfg.device)

    dataloader = getattr(model, f'{cfg.set}_dataloader')()
    print(f'Real batch size is {dataloader.batch_size * cfg.n_samples}')

    name2count = {}
    for i, data in enumerate(tqdm(dataloader)):
        if i % n_jobs != job_id:
            print(f'Skipping batch {i}')
            continue

        new_data = {
            'ligand': TensorDict(**data['ligand']).to(cfg.device),
            'pocket': TensorDict(**data['pocket']).to(cfg.device),
        }
        if 'frag_smiles' in data['ligand']:
            smiles_gt = [s for _ in range(cfg.n_samples) for s in data['ligand']['frag_smiles']]
        else:
            smiles_gt = [None] * cfg.n_samples
        try:
            rdmols, rdpockets, _, x_h_preds, x_h_gts, names, smiles_pred \
                = model.sample(
                    data=new_data,
                    n_samples=cfg.n_samples,
                    sample_with_ground_truth_size=cfg.sample_with_ground_truth_size,
                    return_gt=True,
                    )
        except Exception as e:
            if cfg.set == 'train':
                names = data['ligand']['name']
                print(f'Failed to sample for {names}: {e}')
                continue
            else:
                raise e

        for mol, pocket, x_h_pred, x_h_gt, name, smi_gt, smi_pred in \
            zip(rdmols, rdpockets, x_h_preds, x_h_gts, names, smiles_gt, smiles_pred):

            if model_params['train_params'].use_ev:
                mol = [mol]
            else:
                smi_pred = [s for sublist in smi_pred for s in sublist]
            name = name.replace('.sdf', '')
            idx = name2count.setdefault(name, 0)
            output_dir = Path(samples_dir, name)
            output_dir.mkdir(parents=True, exist_ok=True)

            out_sdf_path = Path(output_dir, f'{idx}_ligand.sdf')
            out_pdb_path = Path(output_dir, f'{idx}_pocket.pdb')
            write_sdf_file(out_sdf_path, mol)
            mols_to_pdbfile([pocket], out_pdb_path)

            # save x and h predictions and ground truth
            torch.save(x_h_pred, Path(output_dir, f'{idx}_x_h_pred.pt'))
            torch.save(x_h_gt, Path(output_dir, f'{idx}_x_h_gt.pt'))

            # save smiles
            smi_pred = '.'.join(smi_pred)
            with open(Path(output_dir, f'{idx}_smiles_gt.txt'), 'w') as f:
                f.write(smi_gt)
            with open(Path(output_dir, f'{idx}_smiles_pred.txt'), 'w') as f:
                f.write(smi_pred)

            name2count[name] += 1


def evaluate(cfg, model_params, samples_dir):
    print('Evaluation...')
    data, table_detailed, table_aggregated, table_detailed_nci = compute_all_metrics(
        in_dir=samples_dir,
        set_name=cfg.set,
        use_ev=model_params['train_params'].use_ev,
        gnina_path=model_params['train_params'].gnina,
        plip_exec=model_params['train_params'].plip,
        reference_path=Path(model_params['train_params'].datadir),
        n_samples=cfg.n_samples,
        exclude_evaluators=[] if cfg.exclude_evaluators is None else cfg.exclude_evaluators,
        trained_dmasif=model_params['predictor_params'].dmasif_params.trained_dmasif if not model_params['connect'] else None,
        frag_encoder_params=cfg.frag_encoder_params if not model_params['connect'] else None,
    )
    with open(Path(samples_dir, 'metrics_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    table_detailed.to_csv(Path(samples_dir, 'metrics_detailed.csv'), index=False)
    table_aggregated.to_csv(Path(samples_dir, 'metrics_aggregated.csv'), index=False)
    table_detailed_nci.to_csv(Path(samples_dir, 'metrics_detailed_frag_nci.csv'), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str)
    p.add_argument('--job_id', type=int, default=0, help='Job ID')
    p.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')
    args = p.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = dict_to_namespace(cfg)

    set_deterministic(seed=cfg.seed)
    disable_rdkit_logging()

    cfg.exclude_evaluators = getattr(cfg, 'exclude_evaluators', None)
    cfg.frag_encoder_params = getattr(cfg, 'frag_encoder_params', None)
    cfg.direct_eval = getattr(cfg, 'direct_eval', False)

    model_params = torch.load(cfg.checkpoint, map_location=cfg.device)['hyper_parameters']
    if 'model_args' in cfg:
        ckpt_args = dict_to_namespace(model_params)
        model_params = combine(ckpt_args, cfg.model_args).__dict__

    ckpt_path = Path(cfg.checkpoint)
    ckpt_name = ckpt_path.parts[-1].split('.')[0]
    n_steps = model_params['simulation_params'].n_steps

    if not cfg.direct_eval:
        samples_dir = Path(cfg.sample_outdir, cfg.set, f'{ckpt_name}_T={n_steps}') or \
                    Path(ckpt_path.parent.parent, 'samples', cfg.set, f'{ckpt_name}_T={n_steps}')
        assert cfg.set in {'val', 'test', 'train'}
        samples_dir.mkdir(parents=True, exist_ok=True)
    else:
        samples_dir = Path(cfg.sample_outdir)

    # save configs
    with open(Path(samples_dir, 'model_params.yaml'), 'w') as f:
        yaml.dump(path_to_str(namespace_to_dict(model_params)), f)
    with open(Path(samples_dir, 'sampling_params.yaml'), 'w') as f:
        yaml.dump(path_to_str(namespace_to_dict(cfg)), f)

    if cfg.sample:
        sample(cfg, model_params, samples_dir, job_id=args.job_id, n_jobs=args.n_jobs)

    if cfg.evaluate:
        assert args.job_id == 0 and args.n_jobs == 1, 'Evaluation is not parallelised on GPU machines'
        evaluate(cfg, model_params, samples_dir)