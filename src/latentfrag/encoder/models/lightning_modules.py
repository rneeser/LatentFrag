from typing import Optional
from pathlib import Path

import json
import torch
import tempfile
import wandb
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.functional.classification import binary_auroc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
import numpy as np
from torch_scatter import scatter
from tqdm import tqdm

from latentfrag.encoder.dmasif.model import dMaSIF
from latentfrag.encoder.dmasif.geometry_processing import atoms_to_points_normals
from latentfrag.encoder.ligand_encoder.gnn import InvariantGraphEncoder
from latentfrag.encoder.ligand_encoder.graphtransformer import GraphTransformer
from latentfrag.encoder.utils.data import prepare_pdb, default_filter, prepare_sdf, TensorDict
from latentfrag.fm.data.msms import pdb_to_points_normals
from latentfrag.encoder.data.pdb_dataset import FragDataset
import latentfrag.encoder.dmasif.geometry_processing as gp
from latentfrag.encoder.models.nci_classifier import NCIClassifier
from latentfrag.encoder.utils.constants import NCIS


class BaseModule(pl.LightningModule):
    def __init__(
            self,
            lr: float,
            batch_size: int,
            num_workers: int,
            dataset_params: dict,
            dataset_class: type,
            surface_parameterization: str,
            surface_params: dict,
            surface_encoder: str,
            surface_encoder_params: dict,
            loss_params: Optional[dict] = None,
            use_ev: bool = False,
            include_esm: bool = False,
            predict_nci: bool = False,
    ):
        super(BaseModule, self).__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.surface_parameterization = surface_parameterization
        self.surface_params = surface_params
        if 'faces' not in self.surface_params:
            self.compute_faces = False
        else:
            self.compute_faces = surface_params['faces']
        self.loss_params = loss_params
        self.surface_encoder_params = surface_encoder_params
        self.use_ev = use_ev
        self.frag_only_loss = loss_params.get('frag_only_loss', None)
        self.include_esm = include_esm
        self.predict_nci = predict_nci

        if surface_encoder == 'dmasif':
            self.surface_encoder = dMaSIF(surface_encoder_params)
        else:
            raise NotImplementedError(f"Surface encoder {surface_encoder} not implemented")

        self.datadir = dataset_params['datadir']
        self.DatasetClass = dataset_class
        if 'pdb_dir' in dataset_params:
            self.pdb_dir = dataset_params['pdb_dir']
        else:
            self.pdb_dir = None
        self.dataset_params = dataset_params
        self.esm_dir =  None
        self.nci_dir = dataset_params['nci_dir'] \
            if 'nci_dir' in dataset_params else None
        self.preprocess = False
        self.processed_dir = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def pdb_to_surface(self, pdb, faces=False):
        faces = self.compute_faces or faces

        # read atomic structure
        protein = prepare_pdb(pdb)

        # Compute surface point cloud
        if self.surface_parameterization == 'dmasif':
            if faces:
                raise NotImplementedError("dMaSIF surfaces do not contain faces")

            if torch.cuda.is_available():
                protein = protein.cuda()

            protein["xyz"], protein["normals"], _ = \
                atoms_to_points_normals(
                    protein["atom_xyz"],
                    protein["batch_atoms"],
                    atomtypes=protein["atomtypes"],
                    resolution=self.surface_params['resolution'],
                    sup_sampling=self.surface_params['sup_sampling'],
                    distance=self.surface_params['distance'],
                )

        elif self.surface_parameterization == 'msms':
            msms_out = pdb_to_points_normals(
                pdb, self.surface_params['msms_bin'],
                self.surface_params['resolution'],
                filter=lambda a: default_filter(a.parent), return_faces=faces)

            protein["xyz"] = torch.from_numpy(msms_out[0]).float()
            protein["normals"] = torch.from_numpy(msms_out[1]).float()
            if faces:
                protein['faces'] = torch.from_numpy(msms_out[2]).float()

        else:
            raise NotImplementedError("Available surface parameterizations "
                                      "are: {'dmasif', 'msms'}")

        return protein

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit' or stage == 'train':
            self.train_dataset = self.DatasetClass(
                datadir=self.datadir,
                split="train",
                surface_fn=self.pdb_to_surface,
                pdb_dir=self.pdb_dir,
                nci_dir=self.nci_dir,
                thresh=float(self.loss_params['thresh']),
                subset=self.dataset_params['subset'],
                use_ev=self.use_ev,
                frag_only_loss=self.frag_only_loss,
                predict_nci=self.predict_nci,
                preprocessing=self.preprocess)

        if stage == 'fit' or stage == 'validate':
            self.val_dataset = self.DatasetClass(
                datadir=self.datadir,
                split="valid",
                surface_fn=self.pdb_to_surface,
                pdb_dir=self.pdb_dir,
                nci_dir=self.nci_dir,
                thresh=float(self.loss_params['thresh']),
                subset=-1,
                use_ev=self.use_ev,
                frag_only_loss=self.frag_only_loss,
                predict_nci=self.predict_nci,
                preprocessing=self.preprocess)

        if stage == 'test':
            self.test_dataset = self.DatasetClass(
                datadir=self.datadir,
                split="test",
                surface_fn=self.pdb_to_surface,
                pdb_dir=self.pdb_dir,
                nci_dir=self.nci_dir,
                thresh=float(self.loss_params['thresh']),
                subset=-1,
                use_ev=self.use_ev,
                frag_only_loss=self.frag_only_loss,
                predict_nci=self.predict_nci,
                preprocessing=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          # persistent_workers=True,
                          collate_fn=self.DatasetClass.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          # persistent_workers=True,
                          collate_fn=self.DatasetClass.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          # persistent_workers=True,
                          collate_fn=self.DatasetClass.collate_fn)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def safe_where(self, tensor, condition, chunk_size=1_000_000_000, depth=0):
        rows = tensor.shape[0]
        positive_idx_1 = []
        positive_idx_2 = []
        distances = []

        for i in range(0, rows, chunk_size):
            chunk = tensor[i:i+chunk_size]
            try:
                chunk_positive = condition(chunk)
                chunk_idx_1, chunk_idx_2 = chunk_positive.nonzero().unbind(1)
                positive_idx_1.append(chunk_idx_1 + i)
                positive_idx_2.append(chunk_idx_2)
                distances.append(chunk[chunk_positive])
            except RuntimeError as e:
                print(f"Error at depth {depth}, chunk_size {chunk_size}: {e}")
                print(f"Tensor shape: {tensor.shape}")
                print(f"Chunk shape: {chunk.shape}")
                if chunk_size > 1:
                    return self.safe_where(tensor, condition, chunk_size // 2, depth + 1)
                else:
                    raise ValueError(f"Failed even with chunk_size 1 at depth {depth}")

        positive_idx_1 = torch.cat(positive_idx_1)
        positive_idx_2 = torch.cat(positive_idx_2)
        distances = torch.cat(distances)
        return positive_idx_1, positive_idx_2, distances


class FragEmbed(BaseModule):
    def __init__(
            self,
            surface_encoder,
            surface_encoder_params,
            ligand_encoder,
            ligand_encoder_params,
            lr,
            batch_size,
            num_workers,
            dataset_params,
            surface_parameterization,
            surface_params,
            loss_params,
            eval_strategies,
    ):
        self.use_ev = ligand_encoder_params.get('use_ev', False)
        self.include_esm = surface_encoder_params.get('include_esm', False) # backward compatibility
        self.predict_nci = loss_params.get('predict_nci', False)
        self.lambda_nci = loss_params.get('lambda_nci', 1.0)
        super(FragEmbed, self).__init__(
            lr, batch_size, num_workers, dataset_params, FragDataset,
            surface_parameterization, surface_params, surface_encoder,
            surface_encoder_params, loss_params, self.use_ev, self.include_esm, self.predict_nci)
        # TODO better saving of hparams; make it not save data directories
        self.save_hyperparameters()

        self.frag_only_loss = loss_params.get('frag_only_loss', None)
        self.lambda_frag = loss_params.get('lambda_frag', 1.0)
        self.lambda_cont = loss_params.get('lambda_cont', 1.0)
        self.intra_frag_margin = loss_params.get('intra_frag_margin', 0.5)

        self.pocket_cutoff = 15.0

        self.eval_frequency = dataset_params.get('eval_frequency', 1)

        assert surface_encoder_params['emb_dims'] == ligand_encoder_params['output_node_nf'], \
            'Embedding dims must match'

        if surface_encoder == 'dmasif':
            assert 'pdb_dir' in dataset_params, 'pdb_dir must be specified'

        if ligand_encoder == 'gnn':
            self.ligand_encoder = InvariantGraphEncoder(**ligand_encoder_params)
        elif ligand_encoder == 'graphtransformer':
            self.ligand_encoder = GraphTransformer(**ligand_encoder_params)
        else:
            raise NotImplementedError(f"{ligand_encoder} ligand encoder not available")

        if self.predict_nci:
            self.nci_classifier = NCIClassifier(surface_encoder_params['emb_dims'], len(NCIS)+1)

        assert 'pdb_dir' in dataset_params , 'pdb_dir must be specified'

        self.loss_type = loss_params.get('loss_type', 'dot')
        self.regularization = loss_params['regularize_desc']
        self.negatives = loss_params['negatives']
        self.thresh = float(loss_params['thresh'])
        self.positive_type = loss_params.get('positive_type', 'binary')
        assert self.thresh < self.pocket_cutoff, f"'Thresh' must be less than {self.pocket_cutoff}A"

        if 'pooled_mol_desc' in loss_params:
            self.pooled_mol_desc = loss_params['pooled_mol_desc']
        else:
            self.pooled_mol_desc = False
        assert sum(self.negatives.values()) == 1.0
        assert self.loss_type in ['dot', 'cos', 'l2']

        self.learned_global = ligand_encoder_params.get('learned_global', False)
        # assert that if learned global also pooled mol desc
        if self.learned_global:
            assert self.pooled_mol_desc, 'If learned global, must also pool mol desc'

        self.curvatures_boosting = None
        self.curvature_scales = surface_encoder_params['curvature_scales']
        self.eval_strategies = eval_strategies

    def compute_curvature_separator(self, n_complexes, test=False):
        all_curvatures = []
        all_labels = []
        examples_cnt = 0

        if test:
            dataloader = self.test_dataloader()
        else:
            dataloader = self.train_dataloader()

        for batch in tqdm(dataloader, total=n_complexes // self.batch_size):
            if examples_cnt >= n_complexes:
                break

            batch = TensorDict(**batch).to(self.device)
            curv_features = gp.curvatures(
                batch["xyz"],
                triangles=None,
                normals=batch["normals"],
                scales=self.curvature_scales,
                batch=batch["batch"],
            )
            surf_xyz, ligand_xyz = batch['xyz'], batch['ligand_xyz']
            surf_batch, ligand_batch = batch['batch'], batch['ligand_batch']
            dists = torch.cdist(surf_xyz, ligand_xyz.float())
            dists = dists + self.thresh * (surf_batch[:, None] != ligand_batch[None, :])

            # Positive and negative examples
            pos_idx = torch.where(dists.min(1).values < self.thresh)[0]
            neg_idx = torch.where(dists.min(1).values >= self.thresh)[0]
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:len(pos_idx)]]
            assert pos_idx.shape == neg_idx.shape

            curvatures = torch.cat([curv_features[pos_idx], curv_features[neg_idx]])
            labels = torch.cat([torch.ones_like(pos_idx), torch.zeros_like(neg_idx)])
            assert curvatures[..., 0].shape == labels.shape

            all_curvatures.append(curvatures)
            all_labels.append(labels)
            examples_cnt += len(batch["batch"].unique())

        all_curvatures = torch.cat(all_curvatures, dim=0)
        all_labels = torch.cat(all_labels, dim=0).squeeze()

        self.curvatures_boosting = HistGradientBoostingClassifier()
        self.curvatures_boosting.fit(all_curvatures.detach().cpu().numpy(), all_labels.detach().cpu().numpy())
        pred_labels = self.curvatures_boosting.predict_proba(all_curvatures.detach().cpu().numpy())[:, 1]
        pred_labels = torch.tensor(pred_labels, device=self.device)
        roc_auc = binary_auroc(pred_labels, all_labels).detach()

        print(f'Prepared GB pocket classifier based on curvature features: ROC-AUC={roc_auc:.3f}')
        print(f'Used {examples_cnt} protein examples')

    def is_concave_like_pocket(self, _curvatures):
        predictions = self.curvatures_boosting.predict(_curvatures.detach().cpu().numpy())
        return torch.tensor(predictions, device=_curvatures.device, dtype=torch.bool)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    def setup(self, stage: Optional[str] = None, n_separator_complexes: int = 750):

        super().setup(stage)

        self.val_strategy = None

        if stage == 'fit':
            self.compute_curvature_separator(n_complexes=n_separator_complexes)
        if stage == 'validate':
            self.val_strategy = [{'pockets': 1.0}]
        if stage == 'test':
            self.compute_curvature_separator(n_complexes=n_separator_complexes, test=True)

    def forward(self, sample):
        # with torch.autocast(device_type='cuda', dtype=torch.float16):  # mixed precision
        surface_embeddings, surface_curvatures = self.surface_encoder(sample)
        ligand_embeddings, ligand_embeddings_global = self.ligand_encoder(
            sample["ligand_xyz"], sample["ligand_types"], sample["ligand_mask"],
            sample["ligand_bonds"], sample["ligand_bond_types"], sample["ligand_mol_feats"], return_global=True)
        neg_ligand_embeddings = None
        neg_liagnd_embeddings_gloabl = None
        if self.frag_only_loss is not None:
            neg_ligand_embeddings, neg_liagnd_embeddings_gloabl = self.ligand_encoder(
                sample["neg_ligand_xyz"], sample["neg_ligand_types"], sample["neg_ligand_mask"],
                sample["neg_ligand_bonds"], sample["neg_ligand_bond_types"], sample["neg_ligand_mol_feats"], return_global=True)
        nci_logits = None
        if self.predict_nci:
            nci_logits = self.nci_classifier(surface_embeddings)
        return surface_embeddings, surface_curvatures, ligand_embeddings, ligand_embeddings_global, neg_ligand_embeddings, neg_liagnd_embeddings_gloabl, nci_logits


    @torch.no_grad()
    def encode_protein(self, pdb_file, faces=False):
        protein = self.pdb_to_surface(pdb_file, faces=faces)
        protein["batch"] = torch.zeros(len(protein['xyz']), dtype=int)
        protein['name'] = pdb_file.name
        protein = protein.to(self.device)
        protein['desc'], _ = self.surface_encoder(protein)
        return protein

    @torch.no_grad()
    def encode_ligand(self, sdf_file):
        ligand = prepare_sdf(sdf_file)
        ligand["batch"] = torch.zeros(len(ligand['xyz']), dtype=int)
        if isinstance(sdf_file, Path):
            ligand['name'] = sdf_file.name
        elif isinstance(sdf_file, str):
            ligand['name'] = sdf_file
        else:
            ligand['name'] = 'ligand'
        ligand = ligand.to(self.device)
        ligand['desc'], ligand['desc_global'] = self.ligand_encoder(
            ligand["xyz"], ligand["types"], ligand["batch"],
            ligand["bonds"], ligand["bond_types"], ligand["mol_feats"], return_global=True)
        return ligand

    def sample_examples(self, surf_xyz, surf_batch, surf_curv,
                        ligand_xyz, ligand_batch, frag_batch, negatives, nci_batch):
        dists = torch.cdist(surf_xyz, ligand_xyz.float())
        dists = dists + self.thresh * (surf_batch[:, None] != ligand_batch[None, :])

        # Positives
        positive_surf_idx, positive_ligand_idx, close_dists = self.safe_where(dists, lambda x: x < self.thresh)
        if self.pooled_mol_desc:
            positive_ligand_idx = frag_batch[positive_ligand_idx]
            # "deduplicate" protein:ligand pairs
            unique_pairs = torch.unique(torch.stack([positive_surf_idx, positive_ligand_idx], dim=1), dim=0)
            positive_surf_idx = unique_pairs[:, 0]
            positive_ligand_idx = unique_pairs[:, 1]
            close_dists = scatter(dists[positive_surf_idx], frag_batch, reduce='min', dim=1)
            close_dists = torch.gather(close_dists, 1, positive_ligand_idx[:, None]).squeeze()
        positive_labels = torch.ones_like(positive_surf_idx)

        # Negatives
        negative_surf_idx, negative_ligand_idx = [], []
        for category, percentage in negatives.items():
            n_negatives_per_category = int(percentage * len(positive_surf_idx))
            if category == 'pockets':  # contact points over batch randomly permuted
                neg_surf_idx = positive_surf_idx[torch.randperm(len(positive_surf_idx))]
                neg_surf_idx = neg_surf_idx[:n_negatives_per_category]
            elif category == 'pockets_fixed':  # contact points permuted over batch with fixed random seed (test overfitting)
                torch.manual_seed(42)
                neg_surf_idx = positive_surf_idx[torch.randperm(len(positive_surf_idx))]
                neg_surf_idx = neg_surf_idx[:n_negatives_per_category]
            elif category == 'rest_concave':
                neg_surf_idx = torch.where(
                    (dists.min(1).values >= self.thresh) & self.is_concave_like_pocket(surf_curv).squeeze()
                )[0]
                if len(neg_surf_idx) > n_negatives_per_category:
                    neg_surf_idx = neg_surf_idx[torch.randint(len(neg_surf_idx), (n_negatives_per_category,))]
            elif category == 'rest_convex':
                neg_surf_idx = torch.where(
                    (dists.min(1).values >= self.thresh) & ~self.is_concave_like_pocket(surf_curv).squeeze()
                )[0]
                if len(neg_surf_idx) > n_negatives_per_category:
                    neg_surf_idx = neg_surf_idx[torch.randint(len(neg_surf_idx), (n_negatives_per_category,))]
            elif category == 'pocket_concave':
                neg_surf_idx = torch.where(
                    (dists.min(1).values >= self.thresh) \
                        & self.is_concave_like_pocket(surf_curv).squeeze() \
                        & (dists.min(1).values < self.pocket_cutoff)
                )[0]
                if len(neg_surf_idx) > n_negatives_per_category:
                    neg_surf_idx = neg_surf_idx[torch.randint(len(neg_surf_idx), (n_negatives_per_category,))]

            elif category == 'pocket_convex':
                neg_surf_idx = torch.where(
                    (dists.min(1).values >= self.thresh) \
                        & ~self.is_concave_like_pocket(surf_curv).squeeze() \
                        & (dists.min(1).values < self.pocket_cutoff)
                )[0]
                if len(neg_surf_idx) > n_negatives_per_category:
                    neg_surf_idx = neg_surf_idx[torch.randint(len(neg_surf_idx), (n_negatives_per_category,))]

            else:
                raise NotImplementedError(category)

            if self.predict_nci:
                non_nci_idx = torch.where((dists.min(1).values < self.pocket_cutoff))[0]
                # remove indices in nci_batch from inidces in non_nci_idx
                non_nci_idx = non_nci_idx[~torch.isin(non_nci_idx, nci_batch)]
            else:
                non_nci_idx = None

            if self.pooled_mol_desc:
                num_units = len(ligand_batch.unique())
            else:
                num_units = len(ligand_xyz)
            neg_ligand_idx = torch.randint(num_units, (len(neg_surf_idx),), device=dists.device)
            negative_surf_idx.append(neg_surf_idx)
            negative_ligand_idx.append(neg_ligand_idx)

        negative_surf_idx = torch.cat(negative_surf_idx, dim=0)
        negative_ligand_idx = torch.cat(negative_ligand_idx, dim=0)
        negative_labels = torch.zeros_like(negative_surf_idx)

        surface_idx = torch.cat([positive_surf_idx, negative_surf_idx])
        ligand_idx = torch.cat([positive_ligand_idx, negative_ligand_idx])
        labels = torch.cat([positive_labels, negative_labels])
        # weights = 1 - (close_dists / self.thresh).clamp(0, 1)
        weights = self.calculate_weights(close_dists, 0.5, self.thresh, 5.5)

        return surface_idx, ligand_idx, labels, non_nci_idx, weights

    def compute_loss(self, surf_xyz, surf_embedding, surf_curv, surf_batch,
                     ligand_xyz, ligand_embedding, ligand_batch, frag_batch,
                     negatives, neg_ligand_embedding, nci_logits, nci_batch, nci_true):

        surface_idx, ligand_idx, labels, non_nci_idx, weights = self.sample_examples(
            surf_xyz=surf_xyz,
            surf_batch=surf_batch,
            surf_curv=surf_curv,
            ligand_xyz=ligand_xyz,
            ligand_batch=ligand_batch,
            frag_batch=frag_batch,
            negatives=negatives,
            nci_batch=nci_batch,
        )

        surf_embedding = surf_embedding[surface_idx]
        ligand_embedding_expand = ligand_embedding[ligand_idx]

        if self.loss_type == 'dot':
            predictions = torch.sum(surf_embedding * ligand_embedding_expand, dim=1)
        elif self.loss_type == 'cos':
            predictions = torch.sum(surf_embedding * ligand_embedding_expand, dim=1)
            predictions = predictions / (surf_embedding ** 2).sum(dim=1).sqrt()
            predictions = predictions / (ligand_embedding_expand ** 2).sum(dim=1).sqrt()
        elif self.loss_type == 'l2':
            predictions = ((surf_embedding - ligand_embedding_expand) ** 2).sum(dim=1).sqrt()
        else:
            raise NotImplementedError

        if self.positive_type == 'binary':
            loss = F.binary_cross_entropy_with_logits(predictions, labels.float())
            contrastive_loss = loss.detach()
        elif self.positive_type == 'regression':
            loss_pos = -F.logsigmoid(predictions[labels == 1]) / weights
            loss_neg = -F.logsigmoid(-predictions[labels == 0])
            loss = (loss_pos.mean() + loss_neg.mean())/2
            contrastive_loss = loss.detach()
        else:
            raise NotImplementedError

        if self.regularization > 0.0:
            l2_reg = (surf_embedding**2).sum() + (ligand_embedding_expand**2).sum()
            l2_reg = l2_reg / (len(surf_embedding) + len(ligand_embedding_expand))
            loss = self.lambda_cont * loss + self.regularization * l2_reg

        neg_frag_loss = None
        neg_frag_cossim = None
        if self.frag_only_loss is not None:
            # get additional loss minimizing the similarity between negative ligands and positive ligands
            cosine_similarity = F.cosine_similarity(neg_ligand_embedding, ligand_embedding, dim=1)
            cosine_similarity = cosine_similarity.mean()
            intra_ligand_loss = torch.relu(cosine_similarity - self.intra_frag_margin)
            loss += self.lambda_frag * intra_ligand_loss
            neg_frag_cossim = cosine_similarity.detach()
            neg_frag_loss = intra_ligand_loss.detach()

        if non_nci_idx is not None:
            nci_logits_interact = nci_logits[nci_batch]
            nci_logits_nointeract = nci_logits[non_nci_idx]
            nci_logits_all = torch.cat([nci_logits_interact, nci_logits_nointeract])
            # add 0 bit to every OHE vector for showing that there is interaction
            nci_true_interact = torch.cat((nci_true, torch.zeros(nci_true.shape[0], 1, device=self.device)), dim=1)
            # no interact have all zeros except the last bit
            nci_true_nointeract = torch.cat((torch.zeros(nci_logits_nointeract.shape[0], nci_true.shape[1]), torch.ones(nci_logits_nointeract.shape[0], 1)), dim=1).to(self.device)
            nci_true_all = torch.cat([nci_true_interact, nci_true_nointeract])
            nci_loss = F.binary_cross_entropy_with_logits(nci_logits_all, nci_true_all)
            loss += self.lambda_nci * nci_loss
            nci_loss = nci_loss.detach()
        else:
            nci_loss = None

        return loss, predictions, labels, contrastive_loss, neg_frag_cossim, neg_frag_loss, nci_loss

    def training_step(self, batch):

        try:
            surf_emb, surf_curv, ligand_emb, ligand_emb_global, neg_lig_emb, neg_lig_emb_global, nci_logits = self.forward(batch)

            if torch.any(torch.isnan(surf_emb)):
                print("Found NaN in surface embedding. Skipping batch...")
                return None

            if torch.any(torch.isnan(ligand_emb)):
                print("Found NaN in ligand embedding. Skipping batch...")
                return None

            self.log('norm_surf_desc/train',
                     torch.linalg.norm(surf_emb, dim=1).mean(),
                     batch_size=len(surf_emb), prog_bar=False)
            if not self.learned_global:
                self.log('norm_lig_desc/train',
                        torch.linalg.norm(ligand_emb, dim=1).mean(),
                        batch_size=len(ligand_emb), prog_bar=False)
            self.log('norm_lig_global_desc/train',
                     torch.linalg.norm(ligand_emb_global, dim=1).mean(),
                     batch_size=len(ligand_emb_global), prog_bar=False)

            loss, preds, labels, contrastive_loss, frag_cossim, frag_loss, nci_loss = self.compute_loss(
                surf_xyz=batch['xyz'],
                surf_embedding=surf_emb,
                surf_curv=surf_curv,
                surf_batch=batch['batch'],
                ligand_xyz=batch['ligand_xyz'],
                ligand_embedding=ligand_emb if not self.pooled_mol_desc else ligand_emb_global,
                ligand_batch=batch['ligand_batch'],
                frag_batch=batch['ligand_mask'],
                negatives=self.negatives,
                neg_ligand_embedding=neg_lig_emb if not self.pooled_mol_desc else neg_lig_emb_global,
                nci_logits=nci_logits,
                nci_batch=batch['nci_mask'] if self.predict_nci else None,
                nci_true=batch['nci_ohe'] if self.predict_nci else None,
            )

        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e

        if torch.isnan(loss):
            print("Loss is NaN. Skipping batch...")
            return None

        # Logging
        batch_size = len(preds)
        self.log('loss/train', loss, batch_size=batch_size)
        roc_auc = binary_auroc(preds, labels)
        self.log('roc_auc/train', roc_auc, batch_size=batch_size, prog_bar=True)
        self.log('contrastive_loss/train', contrastive_loss,
                 batch_size=batch_size, prog_bar=True)
        self.log('pos_desc_dp/train', preds[labels == 1].mean(),
                 batch_size=torch.sum(labels == 1), prog_bar=False)
        self.log('neg_desc_dp/train', preds[labels == 0].mean(),
                 batch_size=torch.sum(labels == 0), prog_bar=False)
        if frag_loss is not None:
            self.log('intra_frag_loss/train', frag_loss, batch_size=batch_size)
            self.log('intra_frag_cossim/train', frag_cossim, batch_size=batch_size)
        if self.predict_nci:
            self.log('nci_loss/train', nci_loss, batch_size=batch_size)

        return {
            'loss': loss,
            'preds': preds,
            'labels': labels
        }

    def _shared_eval(self, batch, prefix, *args):
        with open(self.eval_strategies, 'r') as f:
            strategies = json.load(f)

        loss_negatives = self.negatives

        if self.val_strategy is not None:
            strategies = self.val_strategy
            loss_negatives = strategies[0]

        surf_emb, surf_curv, ligand_emb, ligand_emb_global, neg_lig_emb, neg_lig_emb_global, nci_logits = self.forward(batch)

        for negatives in strategies:
            loss, preds, labels, contrastive_loss, frag_cossim, frag_loss, nci_loss = self.compute_loss(
                surf_xyz=batch['xyz'],
                surf_embedding=surf_emb,
                surf_curv=surf_curv,
                surf_batch=batch['batch'],
                ligand_xyz=batch['ligand_xyz'],
                ligand_embedding=ligand_emb if not self.pooled_mol_desc else ligand_emb_global,
                ligand_batch=batch['ligand_batch'],
                frag_batch=batch['ligand_mask'],
                negatives=negatives,
                neg_ligand_embedding=neg_lig_emb if not self.pooled_mol_desc else neg_lig_emb_global,
                nci_logits=nci_logits,
                nci_batch=batch['nci_mask'] if self.predict_nci else None,
                nci_true=batch['nci_ohe'] if self.predict_nci else None,
            )
            roc_auc = binary_auroc(preds, labels)
            suffix = '+'.join(negatives.keys())
            batch_size = len(preds)
            self.log(f'loss_{suffix}/{prefix}', loss, batch_size=batch_size, sync_dist=True)
            self.log(f'contrastive_loss_{suffix}/{prefix}', contrastive_loss, batch_size=batch_size, sync_dist=True)
            self.log(f'roc_auc_{suffix}/{prefix}', roc_auc, batch_size=batch_size, sync_dist=True)

        loss, preds, labels, contrastive_loss, frag_cossim, frag_loss, nci_loss = self.compute_loss(
            surf_xyz=batch['xyz'],
            surf_embedding=surf_emb,
            surf_curv=surf_curv,
            surf_batch=batch['batch'],
            ligand_xyz=batch['ligand_xyz'],
            ligand_embedding=ligand_emb if not self.pooled_mol_desc else ligand_emb_global,
            ligand_batch=batch['ligand_batch'],
            frag_batch=batch['ligand_mask'],
            negatives=loss_negatives,
            neg_ligand_embedding=neg_lig_emb if not self.pooled_mol_desc else neg_lig_emb_global,
            nci_logits=nci_logits,
            nci_batch=batch['nci_mask'] if self.predict_nci else None,
            nci_true=batch['nci_ohe'] if self.predict_nci else None,
        )

        batch_size = len(preds)
        self.log(f'loss/{prefix}', loss, batch_size=batch_size, sync_dist=True)
        self.log(f'contrastive_loss/{prefix}', contrastive_loss, batch_size=batch_size, sync_dist=True)
        self.log(f'pos_desc_dp/{prefix}', preds[labels == 1].mean(),
                    batch_size=torch.sum(labels == 1), sync_dist=True)
        self.log(f'neg_desc_dp/{prefix}', preds[labels == 0].mean(),
                    batch_size=torch.sum(labels == 0), sync_dist=True)
        if frag_loss is not None:
            self.log(f'intra_frag_loss/{prefix}', frag_loss, batch_size=batch_size)
            self.log(f'intra_frag_cossim/{prefix}', frag_cossim, batch_size=batch_size)
        if self.predict_nci:
            self.log(f'nci_loss/{prefix}', nci_loss, batch_size=batch_size)

        return {
            'loss': loss,
            'preds': preds.detach(),
            'labels': labels.detach(),
        }

    def validation_step(self, batch, *args):
        return self._shared_eval(batch, 'val', *args)

    def test_step(self, batch, *args):
        return self._shared_eval(batch, 'test', *args)

    def compute_metrics(self, step_outputs, prefix):

        if len(step_outputs) < 1:
            return

        # aggregate preds and labels
        preds = torch.cat([x['preds'] for x in step_outputs]).squeeze()
        labels = torch.cat([x['labels'] for x in step_outputs])

        roc_auc = binary_auroc(preds, labels)
        self.log(f'roc_auc/{prefix}', roc_auc, prog_bar=True, sync_dist=True)


    def training_epoch_end(self, training_step_outputs):
        self.compute_metrics(training_step_outputs, 'train')

    def validation_epoch_end(self, validation_step_outputs):
        self.compute_metrics(validation_step_outputs, 'val')
        if self.curvatures_boosting is not None:
            self.compute_pocket_correlation(prefix='val')
        if self.current_epoch % self.eval_frequency == 0:
            self.plot_pca('val')

    def test_epoch_end(self, test_step_outputs):
        self.compute_metrics(test_step_outputs, 'test')
        self.plot_pca('test')
        if self.curvatures_boosting is not None:
            self.compute_pocket_correlation(prefix='test')

    def compute_pocket_correlation(self, prefix, thresh=3.0):
        all_embeddings = []
        all_labels = []
        examples_cnt = 0
        if prefix == 'val':
            loader = self.val_dataloader()
        if prefix == 'test':
            loader = self.test_dataloader()
        for batch in loader:
            batch = TensorDict(**batch).to(self.device)
            embeddings, curv_features = self.surface_encoder(batch)
            if self.current_epoch == 0:
                embeddings = curv_features

            surf_xyz, ligand_xyz = batch['xyz'], batch['ligand_xyz']
            surf_batch, ligand_batch = batch['batch'], batch['ligand_batch']
            dists = torch.cdist(surf_xyz, ligand_xyz.float())
            dists = dists + thresh * (surf_batch[:, None] != ligand_batch[None, :])

            # Positive examples that are concave like pockets
            pos_idx = torch.where(
                (dists.min(1).values < thresh) & ~self.is_concave_like_pocket(curv_features).squeeze()
            )[0]

            # Negative examples that are concave like pockets
            neg_idx = torch.where(
                (dists.min(1).values >= thresh) & ~self.is_concave_like_pocket(curv_features).squeeze()
            )[0]
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:len(pos_idx)]]
            assert pos_idx.shape == neg_idx.shape

            embeddings = torch.cat([embeddings[pos_idx], embeddings[neg_idx]])
            labels = torch.cat([torch.ones_like(pos_idx), torch.zeros_like(neg_idx)])
            assert embeddings[..., 0].shape == labels.shape

            all_embeddings.append(embeddings)
            all_labels.append(labels)
            examples_cnt += len(batch["batch"].unique())

            if examples_cnt > 100:
                break

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0).unsqueeze(-1)

        # Compute least squares
        W = torch.linalg.lstsq(all_embeddings, all_labels.float()).solution
        pred_labels = (all_embeddings @ W)
        mse_loss = F.mse_loss(pred_labels, all_labels).detach()
        mae_loss = F.l1_loss(pred_labels, all_labels).detach()
        bce_loss = F.binary_cross_entropy_with_logits(pred_labels, all_labels.float()).detach()
        roc_auc = binary_auroc(pred_labels, all_labels).detach()

        self.log(f'pocket_MSE/{prefix}', mse_loss, prog_bar=False, sync_dist=True)
        self.log(f'pocket_MAE/{prefix}', mae_loss, prog_bar=False, sync_dist=True)
        self.log(f'pocket_BCE/{prefix}', bce_loss, prog_bar=False, sync_dist=True)
        self.log(f'pocket_ROC_AUC/{prefix}', roc_auc, prog_bar=False, sync_dist=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir, f'pocket_detection_plot.png')
            pred_labels_np = pred_labels.squeeze().detach().cpu().numpy()
            all_labels_np = all_labels.squeeze().detach().cpu().numpy()

            plt.plot()
            plt.hist(pred_labels_np[all_labels_np == 1], bins=100, color='green', alpha=0.3, label='pocket')
            plt.hist(pred_labels_np[all_labels_np == 0], bins=100, color='red', alpha=0.3, label='no pocket')
            plt.title(
                f'Pocket Detection: Least Squares [epoch={self.current_epoch}]\n'
                f'n_proteins={examples_cnt}, ROC-AUC={roc_auc:.3f}'
            )
            plt.xlabel('Curvatures Projection' if self.current_epoch == 0 else 'Embeddings Projection')
            plt.legend()
            plt.tight_layout()
            plt.savefig(tmp_path)
            plt.close()

            wandb.log({f'pocket_HIST/{prefix}': wandb.Image(str(tmp_path))}, commit=True)

    @torch.no_grad()
    def plot_pca(self, prefix):
        self.eval()

        if prefix == 'val':
            loader = self.val_dataloader()
        else:
            loader = self.test_dataloader()

        all_embeddings = []
        n_frags = 0
        for batch in loader:
            batch = TensorDict(**batch).to(self.device)
            _, frag_embeddings = self.ligand_encoder(
                batch["ligand_xyz"], batch["ligand_types"], batch["ligand_mask"],
                batch["ligand_bonds"], batch["ligand_bond_types"],
                batch["ligand_mol_feats"], return_global=True
            )
            all_embeddings.append(frag_embeddings)
            n_frags += frag_embeddings.shape[0]

        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

        pca = PCA(n_components=2)
        pca.fit(all_embeddings)
        pca_embeddings = pca.transform(all_embeddings)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir, f'frag_pca.png')

            plt.plot()
            plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1])
            plt.title(
                f'Fragment embeddings: PCA [epoch={self.current_epoch}]\n'
                f'n_proteins={n_frags}'
            )
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.tight_layout()
            plt.savefig(tmp_path)
            plt.close()

            wandb.log({f'frag_pca/{prefix}': wandb.Image(str(tmp_path))}, commit=True)

    def calculate_weights(self, distances, min_distance, max_distance, sharpness=1.0):
        # Normalize distances to [0, 1] range, accounting for minimum distance
        normalized_distances = (distances - min_distance) / (max_distance - min_distance)
        normalized_distances = normalized_distances.clamp(0, 1)

        # Apply sigmoid function to create a smoother weight distribution
        weights = 1 / (-1 + torch.exp(sharpness * (normalized_distances - 0.5)))

        return weights
