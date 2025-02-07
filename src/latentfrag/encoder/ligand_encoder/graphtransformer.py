############################################################
# Code adapted from RetroBridge by Igashov, Schneuing et al.
# https://github.com/igashov/RetroBridge
############################################################

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch_scatter import scatter as t_scatter
from torch import Tensor

from latentfrag.encoder.ligand_encoder.gt_utils import PlaceHolder
from latentfrag.encoder.ligand_encoder.gt_layers import Xtoy, Etoy, masked_softmax
from latentfrag.encoder.utils.constants import bond_mapping


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E, e_mask1, e_mask2)
        x_y = self.x_y(X, x_mask)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self,
                 n_layers: int,
                 input_dims_node: int,
                 input_dim_edge: int,
                 input_dims_global: int,
                 hidden_mlp_dims: dict,
                 hidden_dims: dict,
                 output_node_nf: int,
                 output_dims_edge: int,
                 output_dims_global: int,
                 act_fn_in: str = 'relu',
                 act_fn_out: str = 'relu',
                 addition=True,
                 project=False,
                 use_ev=False,
                 learned_global=False,):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_node_nf
        self.out_dim_E = output_dims_edge
        self.out_dim_y = output_dims_global
        self.addition = addition
        self.project = project
        self.learned_global = learned_global

        if use_ev:
            # 15 chemical environments of BRICS (no double bonds) + 1 hybridization state
            input_dims_node += (15 + 1)

        if input_dims_node < self.out_dim_X and self.addition and not self.project:
            raise ValueError(f'Output node feature dimension ({self.out_dim_X}) must be \
                             less than or equal to input node feature dimension \
                             ({input_dims_node}) when using addition mode.')
        if input_dims_global < self.out_dim_y and self.addition and self.learned_global and not self.project:
            raise ValueError(f'Output global feature dimension ({self.out_dim_y}) must be \
                             less than or equal to input global feature dimension \
                             ({input_dims_global}) when using addition mode.')
        assert self.out_dim_y > 0 or not self.learned_global, 'Global feature dimension must be > 0.'

        if act_fn_in == 'relu':
            act_fn_in = nn.ReLU()
        else:
            raise ValueError(f'Activation function {act_fn_in} not recognized.')

        if act_fn_out == 'relu':
            act_fn_out = nn.ReLU()
        else:
            raise ValueError(f'Activation function {act_fn_out} not recognized.')

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims_node, hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dim_edge, hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims_global, hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        # self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
        #                                nn.Linear(hidden_mlp_dims['E'], output_dims_edge))

        if self.learned_global:
            self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims_global))
        else:
            self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], self.out_dim_X))


        if self.project:
            # self.project_E = nn.Linear(input_dim_edge, self.out_dim_E)
            if self.learned_global:
                self.project_y = nn.Linear(input_dims_global, self.out_dim_y)
            else:
                self.project_X = nn.Linear(input_dims_node, self.out_dim_X)

    def forward(self,
                coords,
                X_block,
                batch_mask,
                covalent_bonds,
                E_covalent,
                y,
                return_global=False):
        bs = batch_mask.unique().shape[0]
        # get max number of atoms in node_mask for padding
        n_bins = batch_mask.bincount()
        n = n_bins.max()

        # get batched X
        X = torch.zeros(bs, n, X_block.shape[1], device=X_block.device) # bs, n, d
        for i in range(bs):
            X[i, :n_bins[i]] = X_block[batch_mask == i]

        # diag_mask = torch.eye(n)
        # diag_mask = ~diag_mask.type_as(X).bool()
        # diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        E, node_mask = self.get_adj(batch_mask, covalent_bonds, E_covalent, n, bs)

        if self.project:
            # E_to_out = self.project_E(E)
            if self.learned_global:
                y_to_out = self.project_y(y)
            else:
                X_to_out = self.project_X(X)
        else:
            # E_to_out = E[..., :self.out_dim_E]
            if self.learned_global:
                y_to_out = y[..., :self.out_dim_y]
            else:
                X_to_out = X[..., :self.out_dim_X]


        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # E = self.mlp_out_E(E)
        if self.learned_global:
            y = self.mlp_out_y(y)
        else:
            X = self.mlp_out_X(X)

        if self.addition:
            # E = (E + E_to_out)
            if self.learned_global:
                y = (y + y_to_out)
            else:
                X = (X + X_to_out)

        # E = E * diag_mask
        # E = 1/2 * (E + torch.transpose(E, 1, 2))

        # format the output
        if not self.learned_global:
            X = X[node_mask]
        else:
            X = torch.empty(0, device=X.device)

        if return_global and self.learned_global:
            return X, y
        elif return_global and not self.learned_global:
            X_global = t_scatter(X, batch_mask, reduce='mean', dim=0)
            return X, X_global
        else:
            return X

    def get_adj(self, batch_mask, covalent_bonds, bond_types, n, bs):
        """
        Format the adjacency matrix from the input data and batch it.

        :param batch_mask: (m,) mask for the batch (m = # of all atoms)
        :param covalent_bonds: (2, k) indices of all covalent bonds (k = # of covalent bonds)
        :param bond_types: (k, d) one-hot encoding of all covalent bond types
        :param n: int -- number of atoms in the biggest molecule in the batch (n <= m)
        :param bs: int -- batch size (number of molecules in the batch)
        :return: (bs, n, n, d) adjacency matrix (padded to the size of the biggest molecule in the batch)
        """

        # Create the adjacency matrix
        adj = torch.zeros(bs, n, n, bond_types.shape[1], device=covalent_bonds.device)

        # NOBOND bond types
        no_bond = bond_mapping['NOBOND']
        one_hot_no_bond = F.one_hot(torch.tensor([no_bond]), num_classes=len(bond_mapping)).squeeze(0)

        node_mask = torch.zeros(bs, n, device=batch_mask.device)

        # Fill with the covalent bonds
        for i in range(bs):
            mask = batch_mask == i
            num_atoms = mask.sum()
            node_mask[i, :num_atoms] = 1
            mask_covalent = mask[covalent_bonds[0]] & mask[covalent_bonds[1]]
            # add non-covalent bonds where there are atoms
            adj[i, :num_atoms, :num_atoms] = one_hot_no_bond
            if any(mask_covalent):
                shift = covalent_bonds[0][mask_covalent].min()
            else:
                shift = 0
            adj[i, covalent_bonds[0][mask_covalent]-shift, covalent_bonds[1][mask_covalent]-shift] = bond_types[mask_covalent]

        # remove self edges
        adj[:, torch.arange(n), torch.arange(n)] = torch.zeros(bond_types.shape[1], device=adj.device)

        # check if the adjacency matrix is symmetric
        # TODO remove once we're happy with this code
        assert (adj.transpose(1, 2) == adj).all(), 'Adjacency matrix is not symmetric.'

        node_mask = node_mask.bool()

        return adj, node_mask