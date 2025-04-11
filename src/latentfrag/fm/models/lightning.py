'''
Code adapted from:
DrugFlow by A. Schneuing & I. Igashov
https://github.com/LPDI-EPFL/DrugFlow
'''
from typing import Optional
from pathlib import Path
from functools import partial
from itertools import accumulate
import warnings
from argparse import Namespace
import tempfile

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_scatter import scatter_mean
import pytorch_lightning as pl
import numpy as np
from time import time
from rdkit import Chem
from tqdm import tqdm
import pandas as pd

from latentfrag.fm.utils.gen_utils import (Queue,
                                        write_sdf_file,
                                        num_nodes_to_batch_mask,
                                        batch_to_list_for_indices,
                                        batch_to_list,
                                        get_grad_norm,
                                        temp_seed,)
from latentfrag.fm.utils.constants import (atom_encoder,
                                  atom_decoder,
                                  aa_encoder,
                                  aa_decoder,
                                  bond_encoder,
                                  frag_conn_encoder,
                                  frag_conn_decoder,
                                  residue_bond_encoder,
                                  residue_bond_decoder,
                                  aa_atom_index,)
from latentfrag.fm.utils.utils_flows import estimate_angular_scale_factor
from latentfrag.fm.analysis.metrics import CategoricalDistribution, GaussianDistribution
from latentfrag.fm.analysis.sbdd_metrics import FullEvaluator
from latentfrag.fm.analysis.frag_metrics import SimpleFragmentEvaluator
from latentfrag.fm.analysis.evaluate import aggregated_metrics, VALIDITY_METRIC_NAME
from latentfrag.fm.models.dynamics import Dynamics
from latentfrag.fm.models.dynamics_hetero import DynamicsHetero
from latentfrag.fm.models.diffusion_utils import DistributionNodes
from latentfrag.fm.models.flows import (CoordICFM,
                                     FragEmbedICFM,
                                     MarginalFragEmbedICFM,
                                     SphericalICFM,
                                     SO3ICFM)
from latentfrag.fm.models.markov_bridge import UniformPriorMarkovBridge
from latentfrag.fm.data.dataset import ProcessedLigandPocketDataset
from latentfrag.fm.data.data_utils import center_data, repeat_items, TensorDict
from latentfrag.fm.data.coarse2fine import latent2frags, latent2fragcombos, get_candidate_mols, get_predicted_degrees
from latentfrag.fm.analysis.visualization_utils import mols_to_pdbfile, pocket_to_rdkit


def set_default(namespace, key, default_val):
    val = vars(namespace).get(key, default_val)
    setattr(namespace, key, val)


class LatentFrag(pl.LightningModule):
    def __init__(
            self,
            pocket_representation: str,
            train_params: Namespace,
            loss_params: Namespace,
            eval_params: Namespace,
            predictor_params: Namespace,
            simulation_params: Namespace,
            coarse_stage: bool = True,
            rotate: bool = False,
            connect: bool = False,
            debug: bool = False,
            overfit: bool = False,
            const_zt: bool = False,
            trained_encoder_path: Path = None,
    ):
        super(LatentFrag, self).__init__()
        self.save_hyperparameters()

        set_default(train_params, 'use_ev', False)
        set_default(train_params, 'lr', 5.0e-4)
        set_default(train_params, 'clip_grad', True)
        set_default(predictor_params, 'k_neighbours', 10)
        set_default(loss_params, "lambda_x", 1.0)
        set_default(simulation_params, 'h_flow', 'euclidean')
        set_default(train_params, 'plip', None)
        set_default(simulation_params, 'prior_e', 'uniform') # backward compatibility
        set_default(train_params, 'size_histogram', None)

        if trained_encoder_path is not None:
            predictor_params.dmasif_params.trained_dmasif = trained_encoder_path

        assert pocket_representation in {'CA+', 'surface'}
        self.pocket_representation = pocket_representation

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_transform = None
        self.debug = debug
        self.overfit = overfit
        self.const_zt = const_zt
        self.rotate = rotate
        self.connect = connect
        self.coarse_stage = coarse_stage
        # argument required for backward compatibility
        assert self.coarse_stage, 'Only coarse stage implemented'
        self.prefix = 'coarse_'
        self.use_ev = train_params.use_ev

        try:
            self.gnina = train_params.gnina
        except AttributeError:
            self.gnina = None
        self.plip = train_params.plip

        assert not (not self.connect and self.use_ev), 'Cannot use EV without connecting fragments'

        self.reduce = getattr(train_params, 'reduce', None)

        print(f'Pocket representation: {self.pocket_representation}')
        print(f'Rotate: {self.rotate}')
        print(f'Connect fragments: {self.connect}')

        # Training parameters
        self.datadir = train_params.datadir
        if self.datadir is None:
            self.datadir = Path(__file__).parent.parent / 'data'
        self.frag_library = None
        self.fragments_ev = None
        assert Path(train_params.frag_sdfs).exists()

        if not self.use_ev:
            lib_path = Path(train_params.frag_library) / 'library_options_emb2uuid.pt'
        else:
            lib_path = Path(train_params.frag_library) / 'library_ev_unique_emb2uuid.pt'
        assert lib_path.exists(), f'Fragment library not found at {lib_path}'
        self.frag_library = torch.load(lib_path)
        self.fragments_ev = Chem.SDMolSupplier(train_params.frag_sdfs)
        self.fragments_ev = [m for m in self.fragments_ev]

        self.receptor_dir = train_params.datadir
        self.batch_size = train_params.batch_size
        self.lr = train_params.lr
        self.num_workers = train_params.num_workers
        self.clip_grad = train_params.clip_grad
        if self.clip_grad:
            self.gradnorm_queue = Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        # Evaluation parameters
        self.outdir = eval_params.outdir
        self.eval_batch_size = eval_params.eval_batch_size
        self.eval_epochs = eval_params.eval_epochs
        # assert eval_params.visualize_sample_epoch % self.eval_epochs == 0
        self.visualize_sample_epoch = eval_params.visualize_sample_epoch
        self.visualize_chain_epoch = eval_params.visualize_chain_epoch
        self.sample_with_ground_truth_size = eval_params.sample_with_ground_truth_size
        self.n_eval_samples = eval_params.n_eval_samples
        self.n_visualize_samples = eval_params.n_visualize_samples
        self.keep_frames = eval_params.keep_frames
        self.topk = eval_params.topk

        # Feature encoders/decoders
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.coarse_bond_encoder = frag_conn_encoder
        self.coarse_bond_decoder = frag_conn_decoder
        self.aa_encoder = aa_encoder
        self.aa_decoder = aa_decoder
        self.residue_bond_encoder = residue_bond_encoder
        self.residue_bond_decoder = residue_bond_decoder

        if not coarse_stage:
            self.atom_nf = len(self.atom_decoder)
        else:
            self.atom_nf = torch.load(predictor_params.dmasif_params.trained_dmasif)\
                ['hyper_parameters']['surface_encoder_params']['emb_dims']

        self.residue_nf = len(self.aa_decoder)
        if self.pocket_representation == 'CA+':
            self.aa_atom_index = aa_atom_index
            self.n_atom_aa = max([x for aa in aa_atom_index.values() for x in aa.values()]) + 1
            self.residue_nf = (self.residue_nf, self.n_atom_aa)  # (s, V)
        if self.pocket_representation == 'surface':
            self.aa_atom_index = aa_atom_index  # allows visualization and docking
            self.n_atom_aa = max([x for aa in aa_atom_index.values() for x in aa.values()]) + 1
            self.residue_nf = (0, 0)  # (s, V)  # feats get added based on params
        self.coarse_bond_nf = len(self.coarse_bond_decoder)
        self.pocket_bond_nf = len(self.residue_bond_decoder)
        self.x_dim = 3

        # Set up the neural network
        self.T = simulation_params.n_steps
        self.step_size = 1 / self.T
        backbone = predictor_params.backbone
        pocket_encoder = getattr(predictor_params, 'pocket_encoder', 'gvp')

        if pocket_encoder == 'dmasif':
            assert not predictor_params.add_all_atom_diff

        if 'heterogeneous_graph' in predictor_params and predictor_params.heterogeneous_graph:
            self.dynamics = DynamicsHetero(
                atom_nf=self.atom_nf,
                residue_nf=self.residue_nf,
                bond_dict=self.coarse_bond_encoder,
                pocket_bond_nf=len(self.residue_bond_encoder) if pocket_encoder != 'dmasif' else 0,
                model=backbone,
                pocket_node_encoder=pocket_encoder,
                pocket_encoder_params=getattr(predictor_params, pocket_encoder + '_params'),
                model_params=getattr(predictor_params, backbone + '_params'),
                edge_cutoff_ligand=predictor_params.edge_cutoff_ligand,
                edge_cutoff_pocket=predictor_params.edge_cutoff_pocket,
                edge_cutoff_interaction=predictor_params.edge_cutoff_interaction,
                add_spectral_feat=predictor_params.spectral_feat,
                add_nma_feat=predictor_params.normal_modes,
                reflection_equiv=predictor_params.reflection_equivariant,
                d_max=predictor_params.d_max,
                num_rbf=predictor_params.num_rbf,
                self_conditioning=predictor_params.self_conditioning,
                add_all_atom_diff=predictor_params.add_all_atom_diff,
                k_neighbours=predictor_params.k_neighbours,
                predict_frames=self.rotate,
                predict_edges=self.connect,
            )
        else:
            self.dynamics = Dynamics(
                    atom_nf=self.atom_nf,
                    residue_nf=self.residue_nf,
                    joint_nf=predictor_params.joint_nf,
                    bond_dict=self.coarse_bond_encoder,
                    pocket_bond_dict=self.residue_bond_encoder,
                    edge_nf=predictor_params.edge_nf,
                    hidden_nf=predictor_params.hidden_nf,
                    model=backbone,
                    model_params=getattr(predictor_params, backbone + '_params'),
                    edge_cutoff_ligand=predictor_params.edge_cutoff_ligand,
                    edge_cutoff_pocket=predictor_params.edge_cutoff_pocket,
                    edge_cutoff_interaction=predictor_params.edge_cutoff_interaction,
                    add_spectral_feat=predictor_params.spectral_feat,
                    add_nma_feat=predictor_params.normal_modes,
                    self_conditioning=predictor_params.self_conditioning,
                    predict_edges=self.connect,
                )

        histogram_file = Path(self.datadir, f'{self.prefix}size_distribution.npy')
        if histogram_file.exists():
            size_histogram = np.load(histogram_file).tolist()
        elif train_params.size_histogram is not None:
            # relevant when sampling
            size_histogram = np.load(train_params.size_histogram).tolist()
        else:
            size_histogram = None

        self.ligand_frag_type_distribution = None
        self.ligand_atom_type_distribution = None

        frag_emb_dist_file = Path(self.datadir, 'coarse_emb_dist.npy')
        if frag_emb_dist_file.exists():
            frag_emb_dist = np.load(frag_emb_dist_file, allow_pickle=True)
            self.ligand_frag_type_distribution = GaussianDistribution(frag_emb_dist)
        ligand_atom_hist_file = Path(self.datadir, 'ligand_type_histogram.npy')
        if ligand_atom_hist_file.exists():
            ligand_hist = np.load(ligand_atom_hist_file, allow_pickle=True).item()
            self.ligand_atom_type_distribution = CategoricalDistribution(
                ligand_hist, self.atom_encoder)

        # Initialize objects for each variable type
        # coordinates
        self.module_x = CoordICFM(None)

        # nodes
        if simulation_params.h_flow == 'euclidean':
            norm_embs = torch.norm(torch.stack(list(self.frag_library.keys()),dim=1).squeeze(0), dim=-1).mean()
            norm_embs = round(float(norm_embs), 1)
            if simulation_params.prior_h == 'marginal':
                prior_mean_h = self.ligand_frag_type_distribution.mean_p
                prior_std_h = self.ligand_frag_type_distribution.std_p
                self.module_h = MarginalFragEmbedICFM(None, self.atom_nf, norm_embs, prior_mean_h, prior_std_h)
            elif simulation_params.prior_h == 'gaussian':
                self.module_h = FragEmbedICFM(None, self.atom_nf, norm_embs)
        elif simulation_params.h_flow == 'spherical':
            scaling_factor = estimate_angular_scale_factor(torch.stack(list(self.frag_library.keys()),dim=1).squeeze(0))
            self.module_h = SphericalICFM(None, self.atom_nf, scaling_factor)
        else:
            NotImplementedError, f"Unknown flow type {simulation_params.h_flow}"

        # fragment orientation (sample uniformly; no marginal prior)
        self.module_rot = SO3ICFM(None) if self.rotate else None

        # edges
        if self.connect:
            if simulation_params.prior_e == 'uniform':
                self.module_e = UniformPriorMarkovBridge(self.coarse_bond_nf,
                                                        loss_type=loss_params.discrete_loss)

        # distribution of nodes
        if size_histogram is not None:
            self.size_distribution = DistributionNodes(size_histogram)
        else:
            self.size_distribution = None

        # Loss parameters
        self.lambda_x = loss_params.lambda_x
        self.lambda_h = loss_params.lambda_h
        self.lambda_e = loss_params.lambda_e if self.connect else None
        self.lambda_rot = loss_params.lambda_rot if self.rotate else None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 amsgrad=True, weight_decay=1e-12)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                # Path(self.datadir, 'val.pt' if self.debug else 'train.pt'), coarse_stage=self.coarse_stage,
                Path(self.datadir, 'train.pt'), coarse_stage=self.coarse_stage,
                ligand_transform=None, catch_errors=True, pocket_transform=None)
            self.val_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'val.pt'), coarse_stage=self.coarse_stage, ligand_transform=None, pocket_transform=None, unique_pockets=True)
            self.setup_metrics()
        elif stage == 'val':
            self.val_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'val.pt'), coarse_stage=self.coarse_stage, ligand_transform=None, pocket_transform=None, unique_pockets=True)
            self.setup_metrics()
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'test.pt'), coarse_stage=self.coarse_stage, ligand_transform=None, pocket_transform=None, unique_pockets=True)
            self.setup_metrics()
        else:
            raise NotImplementedError

    def setup_metrics(self):
        self.evaluator = FullEvaluator(gnina=self.gnina, plip_exec=self.plip,
                                       exclude_evaluators=['geometry',
                                                           'ring_count',
                                                           'clashes',
                                                           'posebusters'])

        self.frag_evaluator = SimpleFragmentEvaluator()

    def train_dataloader(self):
        shuffle = None if self.overfit or self.debug else True
        return DataLoader(self.train_dataset, self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers,
                          sampler=SubsetRandomSampler([0]) if self.overfit else None,
                          collate_fn = partial(self.train_dataset.collate_fn, prefix=self.prefix,
                                               ligand_transform=self.data_transform),
                          pin_memory=True)

    def val_dataloader(self):
        if self.overfit:
            return self.train_dataloader()

        return DataLoader(self.val_dataset, self.eval_batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=partial(self.val_dataset.collate_fn, prefix=self.prefix,),
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=partial(self.test_dataset.collate_fn, prefix=self.prefix),
                          pin_memory=True)

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def aggregate_metrics(self, step_outputs, prefix):
        if 'timestep' in step_outputs[0]:
            timesteps = torch.cat([x['timestep'] for x in step_outputs]).squeeze()

        if 'loss_per_sample' in step_outputs[0]:
            losses = torch.cat([x['loss_per_sample'] for x in step_outputs])
            pearson_corr = torch.corrcoef(torch.stack([timesteps, losses], dim=0))[0, 1]
            self.log(f'corr_loss_timestep/{prefix}', pearson_corr, prog_bar=False)

        if 'eps_hat_norm' in step_outputs[0]:
            eps_norm = torch.cat([x['eps_hat_norm'] for x in step_outputs])
            pearson_corr = torch.corrcoef(torch.stack([timesteps, eps_norm], dim=0))[0, 1]
            self.log(f'corr_eps_timestep/{prefix}', pearson_corr, prog_bar=False)

    def training_epoch_end(self, training_step_outputs):
        self.aggregate_metrics(training_step_outputs, 'train')

    def compute_loss(self, ligand, pocket, return_info=False):
        """
        Samples time steps and computes network predictions
        """

        # Center sample
        ligand, pocket = center_data(ligand, pocket, self.pocket_representation)
        if self.pocket_representation == 'surface':
            pocket_com = scatter_mean(pocket['x_surface'], pocket['mask_surface'], dim=0)
        else:
            pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # Sample a timestep t for each example in batch
        t = torch.rand(ligand['size'].size(0), device=ligand['x'].device).unsqueeze(-1)

        # Noise
        if self.overfit:
            if self.const_zt:
                seed = 42
                with temp_seed(seed):
                    z0_x = self.module_x.sample_z0(pocket_com, ligand[f'{self.prefix}mask'])
                    z0_h = self.module_h.sample_z0(ligand[f'{self.prefix}mask'])
                    if self.connect:
                        z0_e = self.module_e.sample_z0(ligand[f'{self.prefix}bond_mask'],)
                    if self.rotate:
                        z0_rot = self.module_rot.sample_z0(ligand[f'{self.prefix}mask'])
            else:
                z0_x = self.module_x.sample_z0(pocket_com, ligand[f'{self.prefix}mask'])
                z0_h = self.module_h.sample_z0(ligand[f'{self.prefix}mask'])
                if self.connect:
                    z0_e = self.module_e.sample_z0(ligand[f'{self.prefix}bond_mask'],)
                if self.rotate:
                    z0_rot = self.module_rot.sample_z0(ligand[f'{self.prefix}mask'])
        else:
            z0_x = self.module_x.sample_z0(pocket_com, ligand[f'{self.prefix}mask'])
            z0_h = self.module_h.sample_z0(ligand[f'{self.prefix}mask'])
            if self.connect:
                z0_e = self.module_e.sample_z0(ligand[f'{self.prefix}bond_mask'])
            if self.rotate:
                z0_rot = self.module_rot.sample_z0(ligand[f'{self.prefix}mask'])

        zt_x = self.module_x.sample_zt(z0_x, ligand[f'{self.prefix}x'], t,
                                       ligand[f'{self.prefix}mask'])
        zt_h = self.module_h.sample_zt(z0_h, ligand[f'{self.prefix}one_hot'], t,
                                       ligand[f'{self.prefix}mask'])
        if self.connect:
            zt_e = self.module_e.sample_zt(z0_e, ligand[f'{self.prefix}bond_one_hot'], t,
                                       ligand[f'{self.prefix}bond_mask'])
        else:
            zt_e = None
        if self.rotate:
            z1_rot = ligand['axis_angle'].detach().clone()
            zt_rot = self.module_rot.sample_zt(z0_rot, z1_rot, t, ligand[f'{self.prefix}mask'])
        else:
            z1_rot = None
            zt_rot = None

        # Predict denoising
        bonds_ligand = (ligand[f'{self.prefix}bonds'], zt_e) if self.connect else None
        pred_ligand = self.dynamics(zt_x, zt_h, zt_rot, ligand[f'{self.prefix}mask'], pocket, t,
                                    bonds_ligand=bonds_ligand)

        # Compute L2 loss
        loss_x = self.module_x.compute_loss(pred_ligand['vel'], z0_x,
                                            ligand[f'{self.prefix}x'], t,
                                            ligand[f'{self.prefix}mask'])

        t_next = torch.clamp(t + self.step_size, max=1.0)

        # not logits, keep same name for backward compatibility; L2 loss
        loss_h = self.module_h.compute_loss(pred_ligand['logits_h'], z0_h,
                                            ligand[f'{self.prefix}one_hot'], t,
                                            ligand[f'{self.prefix}mask'],)

        loss = self.lambda_x * loss_x + self.lambda_h * loss_h

        if self.connect:
            loss_e = self.module_e.compute_loss(pred_ligand['logits_e'], zt_e, ligand[f'{self.prefix}bond_one_hot'],
                                                ligand[f'{self.prefix}bond_mask'], t, t_next, bs=ligand[f'{self.prefix}mask'].max() + 1)

            loss = loss + self.lambda_e * loss_e


        if self.rotate:
            loss_rot = self.module_rot.compute_loss(pred_ligand['rot_vec'], z1_rot, zt_rot, t, ligand[f'{self.prefix}mask'])
            loss = loss + self.lambda_rot * loss_rot

        loss = loss.mean(0)

        info = {
            'loss': loss,
            f'loss_{self.prefix}x': loss_x.mean().item(),
            f'loss_{self.prefix}h': loss_h.mean().item(),
        }

        if self.connect:
            info[f'loss_{self.prefix}e'] = loss_e.mean().item()
        if self.rotate:
            info[f'loss_{self.prefix}rot'] = loss_rot.mean().item()

        return (loss, info) if return_info else loss

    def training_step(self, data, *args):
        ligand, pocket = data['ligand'], data['pocket']
        try:
            loss, info = self.compute_loss(ligand, pocket, return_info=True)
        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            if 'No cross edges' in str(e):
                print('WARNING: no cross edges, skipping to the next batch')
                return None
            else:
                raise e

        log_dict = {k: v for k, v in info.items() if isinstance(v, float)
                    or torch.numel(v) <= 1}

        self.log_metrics({'loss': loss, **log_dict}, 'train',
                         batch_size=len(ligand['size']))
        return {f'{self.prefix}loss': loss, **info}

    def validation_step(self, data, *args):
        rdmols, rdpockets, frag_embeddings, x_emb_pred, x_emb_gt, _, smiles = self.sample(
            data=data,
            n_samples=self.n_eval_samples,
            sample_with_ground_truth_size=self.sample_with_ground_truth_size,
            return_gt=True,
        )

        if 'frag_smiles' in data['ligand']:
            smiles_gt = [s for _ in range(self.n_eval_samples) for s in data['ligand']['frag_smiles']]
        else:
            smiles_gt = None

        return {
            'ligands': rdmols,
            'pockets': rdpockets,
            'frag_embeddings': frag_embeddings,
            'x_emb_pred': x_emb_pred,
            'x_emb_gt': x_emb_gt,
            'receptor_files': [Path(self.receptor_dir, 'val', x) for x in data['pocket']['name']],
            'smiles_gt': smiles_gt,
            'smiles_pred': smiles,
        }

    def validation_epoch_end(self, validation_step_outputs):
        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')

        rdmols = [m for x in validation_step_outputs for m in x['ligands']]
        rdpockets = [p for x in validation_step_outputs for p in x['pockets']]
        frag_types = [f for x in validation_step_outputs for f in x['frag_embeddings']]
        frag_types = torch.stack(frag_types, dim=0).cpu().numpy()
        x_emb_pred = [elem for x in validation_step_outputs for elem in x['x_emb_pred']]
        x_emb_gt = [elem for x in validation_step_outputs for elem in x['x_emb_gt']]
        smiles_gt = [s for x in validation_step_outputs for s in x['smiles_gt']]
        smiles_gt = None if all([s is None for s in smiles_gt]) else smiles_gt
        smiles_pred = [s for x in validation_step_outputs for s in x['smiles_pred']]

        if not self.use_ev:
            rdmols_flat = [a for b in rdmols for a in b]
        else:
            rdmols_flat = rdmols

        ligand_atom_types = [atom_encoder[a.GetSymbol()] for m in rdmols_flat for a in m.GetAtoms()]

        ligand_bond_types = []
        for m in rdmols_flat:
            bonds = m.GetBonds()
            no_bonds = m.GetNumAtoms() * (m.GetNumAtoms() - 1) // 2 - m.GetNumBonds()
            ligand_bond_types += [bond_encoder['NOBOND']] * no_bonds
            for b in bonds:
                ligand_bond_types.append(bond_encoder[b.GetBondType().name])

        tic = time()
        results = self.analyze_sample(rdmols, ligand_atom_types, ligand_bond_types,
                                      frag_types, x_emb_pred, x_emb_gt, receptors=rdpockets, smiles_gt=smiles_gt, smiles_pred=smiles_pred)
        self.log_metrics(results, 'val')
        print(f'Evaluation took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0:
            tic = time()

            outdir.mkdir(exist_ok=True, parents=True)

            # center for better visualization
            if not self.use_ev:
                rdmols = [mols[0] for mols in rdmols]
            rdmols = rdmols[:self.n_visualize_samples]
            rdpockets = rdpockets[:self.n_visualize_samples]
            for m, p in zip(rdmols, rdpockets):
                center = m.GetConformer().GetPositions().mean(axis=0)
                for i in range(m.GetNumAtoms()):
                    x, y, z = m.GetConformer().GetPositions()[i] - center
                    m.GetConformer().SetAtomPosition(i, (x, y, z))
                for i in range(p.GetNumAtoms()):
                    x, y, z = p.GetConformer().GetPositions()[i] - center
                    p.GetConformer().SetAtomPosition(i, (x, y, z))

            # save molecule
            write_sdf_file(Path(outdir, 'molecules.sdf'), rdmols)

            # save pocket
            write_sdf_file(Path(outdir, 'pockets.sdf'), rdpockets)

            print(f'Sample visualization took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0:
            tic = time()
            outdir.mkdir(exist_ok=True, parents=True)

            batch = self.val_dataset.collate_fn([
                self.val_dataset[
                    torch.randint(len(self.val_dataset), size=(1,))]
            ])
            batch['pocket'] = batch['pocket'].to(self.device)

            if self.sample_with_ground_truth_size:
                max_size = batch['ligand'][f'{self.prefix}size']

            ligand_chain, pocket_chain, info = self.sample_chain(batch['pocket'], self.keep_frames, max_size)

            # save molecules
            write_sdf_file(Path(outdir, 'chain_ligand.sdf'), ligand_chain)

            # save pocket
            mols_to_pdbfile(pocket_chain, Path(outdir, 'chain_pocket.pdb'))

            self.log_metrics(info, 'val')
            print(f'Chain visualization took {time() - tic:.2f} seconds')

    def analyze_sample(self, rdmols, atom_types, bond_types, frag_types,
                       x_emb_pred, x_emb_gt, receptors=None, smiles_gt=None, smiles_pred=None):
        out = {}

        # Distribution of fragment types
        kl_div_frag = self.ligand_frag_type_distribution.kl_divergence(frag_types) \
            if self.ligand_frag_type_distribution is not None else -1
        out[f'kl_div_{self.prefix}frag_types'] = kl_div_frag

        # Distribution of node types
        kl_div_atom = self.ligand_atom_type_distribution.kl_divergence(atom_types) \
            if self.ligand_atom_type_distribution is not None else -1
        out[f'kl_div_{self.prefix}atom_types'] = kl_div_atom

        if not self.use_ev:
            rdmols_flat = [a for b in rdmols for a in b]
        else:
            rdmols_flat = rdmols

        smiles_pred = [s[0] for s in smiles_pred]

        # Evaluation
        results = []
        if receptors is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                for mol, receptor in zip(tqdm(rdmols_flat, desc='FullEvaluator'), receptors):
                    receptor_path = Path(tmpdir, 'receptor.pdb')
                    Chem.MolToPDBFile(receptor, str(receptor_path))
                    results.append(self.evaluator(mol, receptor_path))
        else:
            for mol in tqdm(rdmols_flat, desc='FullEvaluator'):
                self.evaluator = FullEvaluator(pb_conf='mol')
                results.append(self.evaluator(mol))

        results = pd.DataFrame(results)

        all_dtypes = self.evaluator.dtypes

        # Fragment evaluation
        frag_results = []
        for pred, gt, smile_pred, smile_gt in zip(x_emb_pred, x_emb_gt, smiles_pred, smiles_gt):
            frag_results.append(self.frag_evaluator(pred, gt, smile_pred, smile_gt.split('.')))
        frag_results = pd.DataFrame(frag_results)

        results = pd.concat([results, frag_results], axis=1)

        all_dtypes = {**all_dtypes, **self.frag_evaluator.dtypes}

        agg_results = aggregated_metrics(results, all_dtypes, VALIDITY_METRIC_NAME).fillna(0)
        agg_results['metric'] = agg_results['metric'].str.replace('.', '/')

        out.update(**dict(agg_results[['metric', 'value']].values))

        return out

    def sample_zt_given_zs(self, zs_ligand, pocket, s, t, delta_eps_x=None):

        bonds_ligand = (zs_ligand['bonds'], zs_ligand['e']) if self.connect else None

        pred_ligand = \
            self.dynamics(zs_ligand['x'], zs_ligand['h'], zs_ligand['rot_vec'], zs_ligand['mask'],
                          pocket, s, bonds_ligand=bonds_ligand,
                          )

        if delta_eps_x is not None:
            pred_ligand['vel'] = pred_ligand['vel'] + delta_eps_x

        zt_ligand = zs_ligand.copy()
        zt_ligand['x'] = self.module_x.sample_zt_given_zs(zs_ligand['x'], pred_ligand['vel'], s, t, zs_ligand['mask'])

        zt_ligand['h'] = self.module_h.sample_zt_given_zs(zs_ligand['h'], pred_ligand['logits_h'], s, t, zs_ligand['mask'])

        if self.connect:
            zt_ligand['e'] = self.module_e.sample_zt_given_zs(
                zs_ligand['e'], pred_ligand['logits_e'], s, t, zs_ligand['edge_mask'])
        else:
            zt_ligand['e'] = torch.zeros((len(zs_ligand['edge_mask']), self.coarse_bond_nf), device=zs_ligand['x'].device)

        if self.rotate:
            zt_ligand['rot_vec'] = self.module_rot.sample_zt_given_zs(
                zs_ligand['rot_vec'], pred_ligand['rot_vec'], s, t, zs_ligand['mask'])
        else:
            zt_ligand['rot_vec'] = torch.zeros((len(zs_ligand['mask']), 3), device=zs_ligand['x'].device)

        return zt_ligand

    def simulate(self, ligand, pocket, timesteps, t_start, t_end=1.0, return_frames=1):
        """
        Take a version of the ligand and pocket (at any time step t_start) and
        simulate the generative process from t_start to t_end.
        """

        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        assert 0.0 <= t_start < 1.0
        assert 0 < t_end <= 1.0
        assert t_start < t_end

        device = ligand['x'].device
        n_samples = len(pocket['size'])
        delta_t = (t_end - t_start) / timesteps

        # Initialize output tensors
        out_ligand = {
            'x': torch.zeros((return_frames, len(ligand['mask']), self.x_dim), device=device),
            'h': torch.zeros((return_frames, len(ligand['mask']), self.atom_nf), device=device),
            'e': torch.zeros((return_frames, len(ligand['edge_mask']), self.coarse_bond_nf), device=device),
            'rot_vec': torch.zeros((return_frames, len(ligand['mask']), 3), device=device)
        }
        out_pocket = {
            'x': torch.zeros((return_frames, len(pocket['mask']), 3), device=device),
            'v': torch.zeros((return_frames, len(pocket['mask']), self.n_atom_aa, 3), device=device),
        }


        for i, t in enumerate(torch.linspace(t_start, t_end - delta_t, timesteps)):
            t_array = torch.full((n_samples, 1), fill_value=t, device=device)

            delta_eps_lig = None

            ligand = self.sample_zt_given_zs(
                ligand, pocket, t_array, t_array + delta_t, delta_eps_lig)

            # save frame
            if (i + 1) % (timesteps // return_frames) == 0:
                idx = (i + 1) // (timesteps // return_frames)
                idx = idx - 1

                out_ligand['x'][idx] = ligand['x']
                out_ligand['h'][idx] = ligand['h']
                if self.connect:
                    out_ligand['e'][idx] = ligand['e']
                if self.rotate:
                    out_ligand['rot_vec'][idx] = ligand['rot_vec']

                out_pocket['x'][idx] = pocket['x']
                out_pocket['v'][idx] = pocket['v'][:, :self.n_atom_aa, :]

        # remove frame dimension if only the final molecule is returned
        out_ligand = {k: v.squeeze(0) for k, v in out_ligand.items()}
        out_pocket = {k: v.squeeze(0) for k, v in out_pocket.items()}

        return out_ligand, out_pocket

    def init_ligand(self, num_nodes_lig, pocket):
        device = pocket['x'].device

        n_samples = len(pocket['size'])
        lig_mask = num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)

        # only consider upper triangular matrix for symmetry
        lig_bonds = torch.stack(torch.where(torch.triu(
            lig_mask[:, None] == lig_mask[None, :], diagonal=1)), dim=0)
        lig_edge_mask = lig_mask[lig_bonds[0]]

        # Sample from Normal distribution in the pocket center
        if self.pocket_representation == 'surface':
            pocket_com = scatter_mean(pocket['x_surface'], pocket['mask_surface'], dim=0)
        else:
            pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        z0_x = self.module_x.sample_z0(pocket_com, lig_mask)
        z0_h = self.module_h.sample_z0(lig_mask)
        z0_e = self.module_e.sample_z0(lig_edge_mask) if self.connect else None
        z0_rot = self.module_rot.sample_z0(lig_mask) if self.rotate else None

        return TensorDict(**{
                'x': z0_x, 'h': z0_h, 'e': z0_e, 'mask': lig_mask,
                'bonds': lig_bonds, 'edge_mask': lig_edge_mask,
                'rot_vec': z0_rot
            })

    @torch.no_grad()
    def sample(self, data, n_samples, max_size=None, timesteps=None,
               sample_with_ground_truth_size=False, return_gt=False, **kwargs):
        # TODO transform this into a class that calls the model and samples separately for every module

        timesteps = self.T if timesteps is None else timesteps

        pocket = repeat_items(data['pocket'], n_samples)
        if sample_with_ground_truth_size or return_gt:
            ligand_gt = repeat_items(data['ligand'], n_samples)

        if max_size is None:
            if sample_with_ground_truth_size:
                max_size = ligand_gt[f'{self.prefix}size']
            else:
                max_size = self.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'])

        # Sample from prior
        ligand = self.init_ligand(max_size, pocket)
        ligand['name'] = data['ligand']['name'] * n_samples

        out_tensors_ligand, out_tensors_pocket = self.simulate(
            ligand, pocket, timesteps, 0.0, 1.0)

        # Build mol objects
        x = out_tensors_ligand['x'].detach().cpu()
        ligand_type = out_tensors_ligand['h'].detach().cpu()
        if self.connect:
            edge_type = out_tensors_ligand['e'].argmax(1).detach().cpu()
        else:
            # all non-bonds
            nobond_idx = self.coarse_bond_encoder['NOBOND']
            edge_type = torch.full((ligand['bonds'].size(1),), fill_value=nobond_idx, device='cpu')
        rot = out_tensors_ligand['rot_vec'].detach().cpu()
        lig_mask = ligand['mask'].detach().cpu()
        lig_bonds = ligand['bonds'].detach().cpu()
        lig_edge_mask = ligand['edge_mask'].detach().cpu()
        sizes = torch.unique(ligand['mask'], return_counts=True)[1].tolist()
        offsets = list(accumulate(sizes[:-1]))
        offsets.insert(0, 0)
        offsets = [offset for offset, size in zip(offsets, sizes) if size > 1]
        bonds_list = batch_to_list_for_indices(lig_bonds, lig_edge_mask, offsets)
        bond_ohe_list = batch_to_list(edge_type, lig_edge_mask)
        for i, size in enumerate(sizes):
            if size == 1:
                bonds_list.insert(i, torch.tensor([], device=lig_mask.device))
                bond_ohe_list.insert(i, torch.tensor([], device=lig_mask.device))
        molecules = list(
            zip(batch_to_list(x, lig_mask),
                batch_to_list(ligand_type, lig_mask),
                bonds_list,
                bond_ohe_list,
                batch_to_list(rot, lig_mask)
                )
        )

        if sample_with_ground_truth_size:
            sampled_sizes = torch.tensor([len(m[0]) for m in molecules], device=max_size.device)
            assert torch.allclose(sampled_sizes, ligand_gt[f'{self.prefix}size'])

        # Convert into rdmols
        edge_infos = [get_predicted_degrees(graph[2], graph[3]) for graph in molecules]
        if self.use_ev:
            frag_combos = [latent2frags(
                graph, self.frag_library, self.fragments_ev,
                edge_info[0], self.topk, self.rotate,
                ) for graph, edge_info in zip(molecules, edge_infos)]
        else:
            frag_combos = [latent2fragcombos(
                graph, self.frag_library, self.fragments_ev, self.rotate,
                ) for graph in molecules]

        mol_smi = [get_candidate_mols(
            frag_combo, edge_info[0], edge_info[1]) for frag_combo, edge_info in zip(frag_combos, edge_infos)]
        rdmols = [x[0] for x in mol_smi]
        frag_smiles = [x[1] for x in mol_smi]

        if self.use_ev:
            rdmols = [m[0] for m in rdmols]
            frag_smiles = [s[0] for s in frag_smiles]

        out_pocket = pocket.copy()
        out_pocket['x'] = out_tensors_pocket['x'].detach().cpu()
        out_pocket['v'] = out_tensors_pocket['v'].detach().cpu()
        rdpockets = pocket_to_rdkit(out_pocket, self.pocket_representation,
                                    self.atom_encoder, self.atom_decoder,
                                    self.aa_decoder, self.aa_atom_index)

        # make list of tuples: coarse_x, embedding
        x_emb_pred = []
        x_emb_gt = []

        for graph in molecules:
            x_emb_pred.append((graph[0], graph[1]))

        if return_gt:
            gt_x = batch_to_list(ligand_gt['coarse_x'].detach().cpu(), ligand_gt['coarse_mask'])
            gt_emb = batch_to_list(ligand_gt['coarse_one_hot'].detach().cpu(), ligand_gt['coarse_mask'])
            for x, emb in zip(gt_x, gt_emb):
                x_emb_gt.append((x, emb))
        if return_gt:
            return rdmols, rdpockets, ligand_type, x_emb_pred, x_emb_gt, ligand['name'], frag_smiles
        else:
            return rdmols, rdpockets, ligand_type, x_emb_pred, ligand['name'], frag_smiles

    def sample_chain(self, pocket, keep_frames, max_size=None, timesteps=None, **kwargs):

        info = {}

        timesteps = self.T if timesteps is None else timesteps

        # n_samples = 1
        assert len(pocket['mask'].unique()) == 1, "sample_chain only supports a single sample"

        if max_size is None:
            max_size = self.size_distribution.sample_conditional(n1=None, n2=pocket['size'])

        # Sample from prior
        ligand = self.init_ligand(max_size, pocket)

        out_tensors_ligand, out_tensors_pocket = self.simulate(
            ligand, pocket, timesteps, 0.0, 1.0, return_frames=keep_frames)

        info['traj_displacement_lig'] = torch.norm(out_tensors_ligand['x'][-1] - out_tensors_ligand['x'][0], dim=-1).mean()
        info['traj_rms_lig'] = out_tensors_ligand['x'].std(dim=0).mean()

        def flatten_tensor(chain):
            if len(chain.size()) == 3:  # l=0 values
                return chain.view(-1, chain.size(-1))
            elif len(chain.size()) == 4:  # vectors
                return chain.view(-1, chain.size(-2), chain.size(-1))
            else:
                warnings.warn(f"Could not flatten frame dimension of tensor with shape {list(chain.size())}")
                return chain

        # Flatten
        assert keep_frames == out_tensors_ligand['x'].size(0) == out_tensors_pocket['x'].size(0)
        n_atoms = out_tensors_ligand['x'].size(1)
        n_bonds = out_tensors_ligand['e'].size(1)
        device = out_tensors_ligand['x'].device

        out_tensors_ligand_flat = {k: flatten_tensor(chain) for k, chain in out_tensors_ligand.items()}

        ligand_mask_flat = torch.arange(keep_frames).repeat_interleave(n_atoms).to(device)
        bond_mask_flat = torch.arange(keep_frames).repeat_interleave(n_bonds).to(device)
        edges_flat = ligand['bonds'].repeat(1, keep_frames)

        # Build ligands
        x = out_tensors_ligand_flat['x'].detach().cpu()
        ligand_type = out_tensors_ligand_flat['h'].detach().cpu()
        ligand_mask_flat = ligand_mask_flat.detach().cpu()
        bond_mask_flat = bond_mask_flat.detach().cpu()
        edges_flat = edges_flat.detach().cpu()
        if self.connect:
            edge_type = out_tensors_ligand_flat['e'].argmax(1).detach().cpu()
        else:
            nobond_idx = self.coarse_bond_encoder['NOBOND']
            edge_type = torch.full((n_bonds * keep_frames,), fill_value=nobond_idx, device='cpu')
        rot = out_tensors_ligand_flat['rot_vec'].detach().cpu()
        offsets = torch.zeros(keep_frames, dtype=int)  # edges_flat is already zero-based
        if n_bonds > 1:
            bonds_list = batch_to_list_for_indices(edges_flat, bond_mask_flat, offsets)
            bond_ohe_list = batch_to_list(edge_type, bond_mask_flat)
        else:
            bonds_list = [torch.tensor([]) for _ in range(keep_frames)]
            bond_ohe_list = [torch.tensor([]) for _ in range(keep_frames)]
        molecules = list(
            zip(batch_to_list(x, ligand_mask_flat),
                batch_to_list(ligand_type, ligand_mask_flat),
                bonds_list,
                bond_ohe_list,
                batch_to_list(rot, ligand_mask_flat)
                )
        )

        # Convert into rdmols
        edge_infos = [get_predicted_degrees(graph[2], graph[3]) for graph in molecules]
        if self.use_ev:
            frag_combos = [latent2frags(
                graph, self.frag_library, self.fragments_ev,
                edge_info[0], self.topk, self.rotate,
                ) for graph, edge_info in zip(molecules, edge_infos)]
        else:
            frag_combos = [latent2fragcombos(
                graph, self.frag_library, self.fragments_ev, self.rotate,
                ) for graph in molecules]

        mol_smi = [get_candidate_mols(
            frag_combo, edge_info[0], edge_info[1]) for frag_combo, edge_info in zip(frag_combos, edge_infos)]
        ligand_chain = [x[0] for x in mol_smi]
        ligand_chain = [m[0] for m in ligand_chain]

        # Build pockets
        pocket_chain = pocket_to_rdkit(pocket, self.pocket_representation,
                                       self.atom_encoder, self.atom_decoder,
                                       self.aa_decoder, self.aa_atom_index)

        return ligand_chain, pocket_chain, info

    def configure_gradient_clipping(self, optimizer, optimizer_idx,
                                    gradient_clip_val, gradient_clip_algorithm):

        if not self.clip_grad:
            return

        # Ensure minimum clipping threshold
        min_clip_threshold = 1.0

        # Compute adaptive max_grad_norm with safety checks
        try:
            # Allow gradient norm to be 150% + 2 * stdev of the recent history.
            max_grad_norm = max(
                min_clip_threshold,
                1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
            )
        except (RuntimeError, ZeroDivisionError):
            max_grad_norm = min_clip_threshold

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        try:
            grad_norm = get_grad_norm(params)
        except Exception as e:
            print(f"Error computing grad norm: {e}")
            return

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        # Update gradient norm queue with safety
        current_norm = float(grad_norm) if not torch.isnan(grad_norm) else max_grad_norm
        self.gradnorm_queue.add(min(current_norm, max_grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')

    def compute_zt_traj(self, ligand, pocket, timesteps=100):
        device = ligand['x'].device

        # Center sample
        ligand, pocket = center_data(ligand, pocket, self.pocket_representation)
        if self.pocket_representation == 'surface':
            pocket_com = scatter_mean(pocket['x_surface'], pocket['mask_surface'], dim=0)
        else:
            pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # sample z0
        z0_x = self.module_x.sample_z0(pocket_com, ligand[f'{self.prefix}mask'])
        z0_h = self.module_h.sample_z0(ligand[f'{self.prefix}mask'])
        if self.connect:
            z0_e = self.module_e.sample_z0(ligand[f'{self.prefix}bond_mask'])
        if self.rotate:
            z0_rot = self.module_rot.sample_z0(ligand[f'{self.prefix}mask'])
            z1_rot = ligand['axis_angle'].detach().clone()

        # compute trajectory of zt
        t_start = 0.0
        t_end = 1.0
        n_samples = ligand['size'].shape[0]
        delta_t = (t_end - t_start) / timesteps

        out_ligand = {
            'x': torch.zeros((timesteps + 1, len(ligand[f'{self.prefix}mask']), self.x_dim), device=device),
            'h': torch.zeros((timesteps + 1, len(ligand[f'{self.prefix}mask']), self.atom_nf), device=device),
            'e': torch.zeros((timesteps + 1, len(ligand[f'{self.prefix}bond_mask']), self.coarse_bond_nf), device=device),
            'rot_vec': torch.zeros((timesteps + 1, len(ligand[f'{self.prefix}mask']), 3), device=device)
        }

        z1_h = ligand[f'{self.prefix}one_hot']

        for i, t in enumerate(torch.linspace(t_start, t_end - delta_t, timesteps)):
            t_array = torch.full((n_samples, 1), fill_value=t, device=device)

            zt_x = self.module_x.sample_zt(z0_x, ligand[f'{self.prefix}x'], t_array, ligand[f'{self.prefix}mask'])
            zt_h = self.module_h.sample_zt(z0_h, z1_h, t_array, ligand[f'{self.prefix}mask'])
            if self.connect:
                zt_e = self.module_e.sample_zt(z0_e, ligand[f'{self.prefix}bond_one_hot'], t_array, ligand[f'{self.prefix}bond_mask'])
            else:
                zt_e = torch.zeros((len(ligand[f'{self.prefix}bond_mask']), self.coarse_bond_nf), device=device)
            if self.rotate:
                zt_rot = self.module_rot.sample_zt(z0_rot, z1_rot, t_array, ligand[f'{self.prefix}mask'])
            else:
                zt_rot = torch.zeros((len(ligand[f'{self.prefix}mask']), 3), device=device)

            out_ligand['x'][i] = zt_x
            out_ligand['h'][i] = zt_h
            out_ligand['e'][i] = zt_e
            out_ligand['rot_vec'][i] = zt_rot

        # add z1
        out_ligand['x'][-1] = ligand[f'{self.prefix}x']
        out_ligand['h'][-1] = ligand[f'{self.prefix}one_hot']
        out_ligand['e'][-1] = ligand[f'{self.prefix}bond_one_hot']
        out_ligand['rot_vec'][-1] = ligand['axis_angle']

        return out_ligand

