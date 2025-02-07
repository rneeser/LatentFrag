import argparse
from argparse import Namespace
from pathlib import Path

import torch
from rdkit import Chem

from latentfrag.fm.utils.gen_utils import set_deterministic, disable_rdkit_logging, write_sdf_file
from latentfrag.fm.analysis.visualization_utils import mols_to_pdbfile
from latentfrag.fm.data.data_utils import TensorDict, process_pocket_only, fragment_and_augment
from latentfrag.fm.models.lightning import LatentFrag


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # required parameters
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--pdb', type=Path, required=True, help='Path to PDB file to condition sampling on')
    p.add_argument('--msms_bin', type=str, required=True)
    p.add_argument('--trained_encoder', type=Path, required=True, help='Path to trained encoder model')
    p.add_argument('--frag_library', type=str, required=True)
    p.add_argument('--library_sdf', type=str, required=True)

    # sampling parameters
    p.add_argument('--n_samples', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--n_steps', type=int, default=500)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--use_ev', action='store_true', default=False)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--msms_resolution', type=float, default=1.0)
    p.add_argument('--min_frag_size', type=int, default=8)

    # parameters defining number of nodes generated per sample
    p.add_argument('--num_nodes', type=int, default=None)
    p.add_argument('--size_histogram', type=Path, default=None,
                   help='Path to size histogram to sample number of nodes from. Only relevant if num_nodes is None or not sample_gt_size.')
    p.add_argument('--residue_ids', default=None, nargs="*", type=int,)
    p.add_argument('--sample_gt_size', action='store_true', default=False,
                   help='Sample with ground truth size. If True, num_nodes is ignored. Only possible if providing the reference ligand.')
    p.add_argument('--reference_ligand', type=str, default=None,
                     help='Path to reference ligand for sampling with ground truth size')
    p.add_argument('--datadir', type=Path, default=None,
                   help='Path to data directory containing size histogram that was used for training')

    args = p.parse_args()

    set_deterministic(seed=42)
    disable_rdkit_logging()

    # make output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    chkpt_path = Path(args.checkpoint)
    run_name = chkpt_path.parts[-3]
    chkpt_name = chkpt_path.parts[-1].split('.')[0]
    samples_dir = Path(args.outdir, f'{chkpt_name}_T={args.n_steps}')

    # assert that either datadir
    if args.datadir is not None:
        size_histogram_fn = Path(args.datadir, 'coarse_size_distribution.npy')
        if size_histogram_fn.exists():
            hist_exists = True
        else:
            hist_exists = False
    else:
        hist_exists = False

    assert args.num_nodes is not None or args.sample_gt_size or hist_exists, \
        'Either provide num_nodes, sample_gt_size or a size histogram.'
    assert args.reference_ligand is not None or not args.sample_gt_size, \
        'Provide reference ligand for sampling with ground truth size.'

    train_params = {
        'datadir': args.datadir,
        'frag_library': args.frag_library,
        'frag_sdfs': args.library_sdf,
        'use_ev': args.use_ev,
        'num_workers': args.num_workers,
        'batch_size': args.batch_size,
        'size_histogram': args.size_histogram,
    }
    train_params = Namespace(**train_params)

    model = LatentFrag.load_from_checkpoint(args.checkpoint,
                                            map_location=args.device,
                                            train_params=train_params,
                                            trained_encoder_path=args.trained_encoder)

    model.batch_size = model.eval_batch_size = args.batch_size
    model.eval().to(args.device)
    print(f'Real batch size is {args.batch_size * args.n_samples}')

    name2count = {}

    if args.reference_ligand is not None:
        reference_ligand = Chem.SDMolSupplier(args.reference_ligand)[0]

    ligand_coords = None
    if args.residue_ids is None and args.reference_ligand is not None:
        ligand_coords = torch.tensor(reference_ligand.GetConformer().GetPositions(),
                                     dtype=torch.float32)

    # process protein
    pocket = process_pocket_only(pdb_file = args.pdb,
                                 msms_bin = args.msms_bin,
                                 msms_resolution = args.msms_resolution,
                                 pocket_resids = args.residue_ids,
                                 ligand_coords = ligand_coords,)

    if args.reference_ligand is not None and args.sample_gt_size:
        combos, fragments, frags_ev, atom_mapping = fragment_and_augment(reference_ligand, args.min_frag_size)
        num_nodes = len(combos[0])
    elif args.num_nodes is not None:
        num_nodes = args.num_nodes
    else:
        num_nodes = None

    data = {
        'ligand': {
            'name': [args.pdb.name.replace('.pdb', '_sample')],
        },
        'pocket': {
            'name': [args.pdb.name],
        },
    }

    # add processed pocket
    data['pocket'].update(pocket)

    new_data = {
        'ligand': TensorDict(**data['ligand']).to(args.device),
        'pocket': TensorDict(**data['pocket']).to(args.device),
    }

    if args.num_nodes is not None:
        print(f'Sampling {args.num_nodes} nodes/fragments each.')
    elif args.sample_gt_size:
         print(f'Sampling with ground truth size: {args.sample_gt_size}')
         print(f'Sampling {num_nodes} nodes/fragments each.')
    else:
        print(f'Sampling from size histogram.')

    print(f'Sampling n={args.n_samples} for {args.pdb.name}...')
    rdmols, rdpockets, _, x_h_preds, names, smiles_pred = \
        model.sample(new_data,
                     n_samples=args.n_samples,
                     timesteps=args.n_steps,
                     max_size=num_nodes,)
    print('Finished sampling')

    for mol, pocket, x_h_pred, name, smi_pred in \
        zip(rdmols, rdpockets, x_h_preds, names, smiles_pred):

        if args.use_ev:
                mol = [mol]
        else:
            smi_pred = [s for sublist in smi_pred for s in sublist]

        name = name.replace('.sdf', '')
        output_dir = Path(samples_dir, name)
        output_dir.mkdir(parents=True, exist_ok=True)

        idx = name2count.setdefault(name, 0)
        out_sdf_path = Path(output_dir, f'{idx}_ligand.sdf')
        out_pdb_path = Path(output_dir, f'{idx}_pocket.pdb')
        write_sdf_file(out_sdf_path, mol)
        mols_to_pdbfile([pocket], out_pdb_path)

        torch.save(x_h_pred, Path(output_dir, f'{idx}_x_h_pred.pt'))

        smi_pred = '.'.join(smi_pred)
        with open(Path(output_dir, f'{idx}_smiles_pred.txt'), 'w') as f:
            f.write(smi_pred)

        name2count[name] += 1
