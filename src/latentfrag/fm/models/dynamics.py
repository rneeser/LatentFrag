from collections.abc import Iterable
from abc import abstractmethod
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from latentfrag.fm.utils.constants import INT_TYPE, FLOAT_TYPE
from latentfrag.fm.models.gvp import GVPModel, GVP, LayerNorm


class DynamicsBase(nn.Module):
    """
    Implements self-conditioning logic and basic functions
    """
    def __init__(
            self,
            predict_frames=False,
            add_spectral_feat=False,
            self_conditioning=False,
            predict_edges=True,
    ):
        super().__init__()

        if not hasattr(self, 'predict_frames'):
            self.predict_frames = predict_frames

        if not hasattr(self, 'add_spectral_feat'):
            self.add_spectral_feat = add_spectral_feat

        if not hasattr(self, 'self_conditioning'):
            self.self_conditioning = self_conditioning

        if not hasattr(self, 'predict_edges'):
            self.predict_edges = predict_edges

        if self.self_conditioning:
            self.prev_ligand = None

    @abstractmethod
    def _forward(self, x_atoms, h_atoms, rot_vec, mask_atoms, pocket, t, bonds_ligand=None,
                 h_atoms_sc=None, e_atoms_sc=None):
        """
        Implement forward pass.
        Returns:
            - vel
            - h_final_atoms
            - edge_final_atoms
            - fragment_rot
        """
        pass

    def make_sc_input(self, pred_ligand):

        h_atoms_sc = (pred_ligand['logits_h'], pred_ligand['vel'].unsqueeze(1))

        if self.predict_edges:
            e_atoms_sc = pred_ligand['logits_e']
        else:
            e_atoms_sc = None

        if self.predict_frames:
            h_atoms_sc = (torch.cat([h_atoms_sc[0], pred_ligand['rot_vec']], dim=-1), h_atoms_sc[1])

        return h_atoms_sc, e_atoms_sc

    def forward(self, x_atoms, h_atoms, rot_vec, mask_atoms, pocket, t, bonds_ligand=None,):
        """
        Implements self-conditioning as in https://arxiv.org/abs/2208.04202
        """

        h_atoms_sc, e_atoms_sc = None, None

        h_shape = h_atoms.shape

        if self.predict_frames:
            # concat rot_vec to h_atoms
            h_atoms = torch.cat([h_atoms, rot_vec], dim=-1)

        if self.self_conditioning:

            # Sampling: use previous prediction in all but the first time step
            if not self.training and t.min() > 0.0:
                assert t.min() == t.max(), "currently only supports sampling at same time steps"
                assert self.prev_ligand is not None

            else:
                # Create zero tensors
                zeros_ligand = {'logits_h': torch.zeros(h_shape, device=h_atoms.device),
                                'vel': torch.zeros_like(x_atoms)}

                if self.predict_edges:
                    zeros_ligand['logits_e'] = torch.zeros_like(bonds_ligand[1])

                if self.predict_frames:
                    zeros_ligand['rot_vec'] = torch.zeros_like((rot_vec))

                # Training: use 50% zeros and 50% predictions with detached gradients
                if self.training and random.random() > 0.5:
                    with torch.no_grad():
                        h_atoms_sc, e_atoms_sc = self.make_sc_input(zeros_ligand)

                        self.prev_ligand = self._forward(
                            x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand,
                            h_atoms_sc, e_atoms_sc)

                # use zeros for first sampling step and 50% of training
                else:
                    self.prev_ligand = zeros_ligand

            h_atoms_sc, e_atoms_sc = self.make_sc_input(self.prev_ligand)

        pred_ligand = self._forward(
            x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand,
            h_atoms_sc, e_atoms_sc
        )

        if self.self_conditioning and not self.training:
            self.prev_ligand = pred_ligand.copy()

        return pred_ligand

    def compute_extra_features(self, batch_mask, edge_indices, edge_types):

        feat = torch.zeros(len(batch_mask), 0, device=batch_mask.device)

        if not self.add_spectral_feat:
            return feat

        adj = batch_mask[:, None] == batch_mask[None, :]

        E = torch.zeros_like(adj, dtype=INT_TYPE)
        E[edge_indices[0], edge_indices[1]] = edge_types

        A = (E > 0).float()

        if self.add_spectral_feat:
            feat = torch.cat([feat, eigenfeatures(A, batch_mask)], dim=-1)

        return feat


class Dynamics(DynamicsBase):
    def __init__(self,
                 atom_nf,
                 residue_nf,
                 joint_nf,
                 bond_dict,
                 pocket_bond_dict,
                 edge_nf,
                 hidden_nf,
                 act_fn=torch.nn.SiLU(),
                 condition_time=True,
                 model='gvp',
                 model_params=None,
                 edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None,
                 edge_cutoff_interaction=None,
                 add_spectral_feat=False,
                 add_nma_feat=False,
                 self_conditioning=False,
                 predict_edges=True,):
        super().__init__()
        self.model = model
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.hidden_nf = hidden_nf
        self.bond_dict = bond_dict
        self.pocket_bond_dict = pocket_bond_dict
        if predict_edges:
            self.bond_nf = len(bond_dict)
        else:
            self.bond_nf = 0
        self.pocket_bond_nf = len(pocket_bond_dict)
        self.edge_nf = edge_nf
        self.add_spectral_feat = add_spectral_feat
        self.add_nma_feat = add_nma_feat
        self.self_conditioning = self_conditioning
        self.predict_edges = predict_edges

        if self.self_conditioning:
            self.prev_vel = None
            self.prev_h = None
            self.prev_e = None
            self.prev_a = None

        lig_nf = atom_nf
        if self.add_spectral_feat:
            lig_nf = lig_nf + 5

        if not isinstance(joint_nf, Iterable):
            # joint_nf contains only scalars
            joint_nf = (joint_nf, 0)


        if isinstance(residue_nf, Iterable):
            _atom_in_nf = (lig_nf, 0)

            if self.add_nma_feat:
                residue_nf = (residue_nf[0], residue_nf[1] + 5)

            if self.self_conditioning:
                _atom_in_nf = (_atom_in_nf[0] + atom_nf, 1)

            self.atom_encoder = nn.Sequential(
                GVP(_atom_in_nf, joint_nf, activations=(act_fn, torch.sigmoid)),
                LayerNorm(joint_nf, learnable_vector_weight=True),
                GVP(joint_nf, joint_nf, activations=(None, None)),
            )

            self.residue_encoder = nn.Sequential(
                GVP(residue_nf, joint_nf, activations=(act_fn, torch.sigmoid)),
                LayerNorm(joint_nf, learnable_vector_weight=True),
                GVP(joint_nf, joint_nf, activations=(None, None)),
            )

        else:
            # No vector-valued input features
            assert joint_nf[1] == 0

            # self-conditioning not yet supported
            assert not self.self_conditioning

            # Normal mode features are vectors
            assert not self.add_nma_feat

            self.atom_encoder = nn.Sequential(
                nn.Linear(lig_nf, 2 * atom_nf),
                act_fn,
                nn.Linear(2 * atom_nf, joint_nf[0])
            )

            self.residue_encoder = nn.Sequential(
                nn.Linear(residue_nf, 2 * residue_nf),
                act_fn,
                nn.Linear(2 * residue_nf, joint_nf[0])
            )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf[0], 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )

        if predict_edges:
            self.edge_decoder = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, self.bond_nf)
            )

        _atom_bond_nf = 2 * self.bond_nf if self.self_conditioning and predict_edges else self.bond_nf
        self.ligand_bond_encoder = nn.Sequential(
            nn.Linear(_atom_bond_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.edge_nf)
        )

        self.pocket_bond_encoder = nn.Sequential(
            nn.Linear(self.pocket_bond_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.edge_nf)
        )

        # self.ligand_nobond_emb = nn.Parameter(torch.zeros(self.edge_nf))
        # self.pocket_nobond_emb = nn.Parameter(torch.zeros(self.edge_nf))
        self.cross_emb = nn.Parameter(torch.zeros(self.edge_nf),
                                      requires_grad=True)

        if condition_time:
            dynamics_node_nf = (joint_nf[0] + 1, joint_nf[1])
        else:
            print('Warning: dynamics model is NOT conditioned on time.')
            dynamics_node_nf = (joint_nf[0], joint_nf[1])

        if model == 'gvp':
            self.net = GVPModel(
                node_in_dim=dynamics_node_nf,
                node_h_dim=model_params.node_h_dim,
                node_out_nf=joint_nf[0],
                edge_in_nf=self.edge_nf,
                edge_h_dim=model_params.edge_h_dim,
                edge_out_nf=hidden_nf,
                num_layers=model_params.n_layers,
                drop_rate=model_params.dropout,
                vector_gate=model_params.vector_gate,
                reflection_equiv=model_params.reflection_equivariant,
                d_max=model_params.d_max,
                num_rbf=model_params.num_rbf,
                update_edge_attr=True
            )
        else:
            raise NotImplementedError(f"{model} is not available")

        # self.device = device
        # self.n_dims = n_dims
        self.condition_time = condition_time

    def _forward(self, x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand=None,
                h_atoms_sc=None, e_atoms_sc=None, h_residues_sc=None):
        """
        :param x_atoms:
        :param h_atoms:
        :param mask_atoms:
        :param pocket: must contain keys: 'x', 'one_hot', 'mask', 'bonds' and 'bond_one_hot'
        :param t:
        :param bonds_ligand: tuple - bond indices (2, n_bonds) & bond types (n_bonds, bond_nf)
        :param h_atoms_sc: additional node feature for self-conditioning, (s, V)
        :param e_atoms_sc: additional edge feature for self-conditioning, only scalar
        :param h_residues_sc: additional node feature for self-conditioning, tensor or tuple
        :return:
        """
        x_residues, h_residues, mask_residues = pocket['x'], pocket['one_hot'], pocket['mask']
        if 'bonds' in pocket:
            bonds_pocket = (pocket['bonds'], pocket['bond_one_hot'])
        else:
            bonds_pocket = None

        if 'v' in pocket:
            v_residues = pocket['v']
            if self.add_nma_feat:
                v_residues = torch.cat([v_residues, pocket['nma_vec']], dim=1)
            h_residues = (h_residues, v_residues)

        if h_residues_sc is not None:
            h_residues = (torch.cat([h_residues[0], h_residues_sc], dim=-1), h_residues[1])

        # get graph edges and edge attributes
        if bonds_ligand is not None:
            # NOTE: 'bond' denotes one-directional edges and 'edge' means bi-directional

            # make sure messages are passed both ways
            ligand_edge_indices = torch.cat(
                [bonds_ligand[0], bonds_ligand[0].flip(dims=[0])], dim=1)
            ligand_edge_types = torch.cat([bonds_ligand[1], bonds_ligand[1]], dim=0)

            # add auxiliary features to ligand nodes
            extra_features = self.compute_extra_features(
                mask_atoms, ligand_edge_indices, ligand_edge_types.argmax(-1))
            h_atoms = torch.cat([h_atoms, extra_features], dim=-1)
        else:
            ligand_edge_indices = None
            ligand_edge_types = None

        if bonds_pocket is not None:
            # make sure messages are passed both ways
            pocket_edge_indices = torch.cat(
                [bonds_pocket[0], bonds_pocket[0].flip(dims=[0])], dim=1)
            pocket_edge_types = torch.cat([bonds_pocket[1], bonds_pocket[1]], dim=0)

        if h_atoms_sc is not None:
            h_atoms = (torch.cat([h_atoms, h_atoms_sc[0]], dim=-1),
                       h_atoms_sc[1])

        if e_atoms_sc is not None:
            e_atoms_sc = torch.cat([e_atoms_sc, e_atoms_sc], dim=0)
            ligand_edge_types = torch.cat([ligand_edge_types, e_atoms_sc], dim=-1)

        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)
        e_ligand = self.ligand_bond_encoder(ligand_edge_types)
        e_pocket = self.pocket_bond_encoder(pocket_edge_types)

        if isinstance(h_atoms, tuple):
            h_atoms, v_atoms = h_atoms
            h_residues, v_residues = h_residues
            v = torch.cat((v_atoms, v_residues), dim=0)
        else:
            v = None

        edges, edge_feat = self.get_edges(
            mask_atoms, mask_residues, x_atoms, x_residues,
            bond_inds_ligand=ligand_edge_indices, bond_inds_pocket=pocket_edge_indices,
            bond_feat_ligand=e_ligand, bond_feat_pocket=e_pocket)

        # combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        assert torch.all(mask[edges[0]] == mask[edges[1]])

        if self.model == 'gvp':
            h_final, vel, edge_final = self.net(
                h, x, edges, v=v, batch_mask=mask, edge_attr=edge_feat)
        else:
            raise NotImplementedError(f"Wrong model ({self.model})")

        # decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final_atoms)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                h_final_atoms[torch.isnan(h_final_atoms)] = 0.0
            else:
                raise ValueError("NaN detected in network output")

        # TODO add rot
        pred_ligand = {'vel': vel[:len(mask_atoms)],
                       'logits_h': h_final_atoms}

        if self.predict_edges:
            # predict edge type
            ligand_edge_mask = (edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))
            edge_final = edge_final[ligand_edge_mask]
            edges = edges[:, ligand_edge_mask]

            # Symmetrize
            edge_logits = torch.zeros(
                (len(mask_atoms), len(mask_atoms), self.hidden_nf),
                device=mask_atoms.device)
            edge_logits[edges[0], edges[1]] = edge_final
            edge_logits = (edge_logits + edge_logits.transpose(0, 1)) * 0.5

            # return upper triangular elements only (matching the input)
            edge_logits = edge_logits[bonds_ligand[0][0], bonds_ligand[0][1]]

            edge_final_atoms = self.edge_decoder(edge_logits)

            pred_ligand['logits_e'] = edge_final_atoms

        return pred_ligand

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand,
                  x_pocket, bond_inds_ligand=None, bond_inds_pocket=None,
                  bond_feat_ligand=None, bond_feat_pocket=None, self_edges=False):

        # Adjacency matrix
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

            # Add missing bonds if they got removed
            adj_ligand[bond_inds_ligand[0], bond_inds_ligand[1]] = True

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

            # Add missing bonds if they got removed
            adj_pocket[bond_inds_pocket[0], bond_inds_pocket[1]] = True

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)

        if not self_edges:
            adj = adj ^ torch.eye(*adj.size(), out=torch.empty_like(adj))

        # # ensure that edge definition is consistent if bonds are provided (for loss computation)
        # if bond_inds_ligand is not None:
        #     # remove ligand edges
        #     adj[:adj_ligand.size(0), :adj_ligand.size(1)] = False
        #     edges = torch.stack(torch.where(adj), dim=0)
        #     # add ligand edges back with original definition
        #     edges = torch.cat([bond_inds_ligand, edges], dim=-1)
        # else:
        #     edges = torch.stack(torch.where(adj), dim=0)

        # Feature matrix
        ligand_nobond_onehot = F.one_hot(torch.tensor(
            self.bond_dict['NOBOND'], device=bond_feat_ligand.device),
            num_classes=self.ligand_bond_encoder[0].in_features)
        ligand_nobond_emb = self.ligand_bond_encoder(
            ligand_nobond_onehot.to(FLOAT_TYPE))
        feat_ligand = ligand_nobond_emb.repeat(*adj_ligand.shape, 1)
        feat_ligand[bond_inds_ligand[0], bond_inds_ligand[1]] = bond_feat_ligand

        pocket_nobond_onehot = F.one_hot(torch.tensor(
            self.pocket_bond_dict['NOBOND'], device=bond_feat_pocket.device),
            num_classes=self.pocket_bond_nf)
        pocket_nobond_emb = self.pocket_bond_encoder(
            pocket_nobond_onehot.to(FLOAT_TYPE))
        feat_pocket = pocket_nobond_emb.repeat(*adj_pocket.shape, 1)
        feat_pocket[bond_inds_pocket[0], bond_inds_pocket[1]] = bond_feat_pocket

        feat_cross = self.cross_emb.repeat(*adj_cross.shape, 1)

        feats = torch.cat((torch.cat((feat_ligand, feat_cross), dim=1),
                           torch.cat((feat_cross.transpose(0, 1), feat_pocket), dim=1)), dim=0)

        # Return results
        edges = torch.stack(torch.where(adj), dim=0)
        edge_feat = feats[edges[0], edges[1]]

        return edges, edge_feat


def binomial_coefficient(n, k):
    # source: https://discuss.pytorch.org/t/n-choose-k-function/121974
    return ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()).exp()


# TODO: also consider directional aggregation as in:
#  Beaini, Dominique, et al. "Directional graph networks."
#  International Conference on Machine Learning. PMLR, 2021.
def eigenfeatures(A, batch_mask, k=5):

    # TODO, see:
    # - https://github.com/cvignac/DiGress/blob/main/src/diffusion/extra_features.py
    # - https://arxiv.org/pdf/2209.14734.pdf (Appendix B.2)

    # split adjacency matrix
    batch = []
    for i in torch.unique(batch_mask, sorted=True):  # TODO: optimize (try to avoid loop)
        batch_inds = torch.where(batch_mask == i)[0]
        batch.append(A[torch.meshgrid(batch_inds, batch_inds, indexing='ij')])

    eigenfeats = [get_nontrivial_eigenvectors(adj)[:, :k] for adj in batch]
    # if there are less than k non-trivial eigenvectors
    eigenfeats = [torch.cat([
        x, torch.zeros(x.size(0), max(k - x.size(1), 0), device=x.device)], dim=-1)
        for x in eigenfeats]
    return torch.cat(eigenfeats, dim=0)


def get_nontrivial_eigenvectors(A, normalize_l=True, thresh=1e-5,
                                norm_eps=1e-12):
    """
    Compute eigenvectors of the graph Laplacian corresponding to non-zero
    eigenvalues.
    """
    assert (A == A.T).all(), "undirected graph"

    # Compute laplacian
    d = A.sum(-1)
    D = d.diag()
    L = D - A

    if normalize_l:
        D_inv_sqrt = (1 / (d.sqrt() + norm_eps)).diag()
        L = D_inv_sqrt @ L @ D_inv_sqrt

    # Eigendecomposition
    # eigenvalues are sorted in ascending order
    # eigvecs matrix contains eigenvectors as its columns
    eigvals, eigvecs = torch.linalg.eigh(L)

    # index of first non-trivial eigenvector
    try:
        idx = torch.nonzero(eigvals > thresh)[0].item()
    except IndexError:
        # recover if no non-trivial eigenvectors are found
        idx = eigvecs.size(1)

    return eigvecs[:, idx:]
