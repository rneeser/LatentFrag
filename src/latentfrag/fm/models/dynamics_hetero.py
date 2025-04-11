'''
Code adapted from:
DrugFlow by A. Schneuing & I. Igashov
https://github.com/LPDI-EPFL/DrugFlow
'''
import argparse
from collections.abc import Iterable
from collections import defaultdict
from functools import partial
import functools
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.utils.hetero import check_add_self_loops
from torch_geometric.nn.conv.hgt_conv import group

from latentfrag.fm.models.dynamics import DynamicsBase
from latentfrag.fm.models import gvp
from latentfrag.fm.models.gvp import (GVP,
                                      _rbf,
                                      _normalize,
                                      tuple_index,
                                      tuple_sum,
                                      _split,
                                      tuple_cat,
                                      _merge)
from latentfrag.encoder.dmasif.model import dMaSIF, knn_atoms
from latentfrag.fm.utils.gen_utils import process_in_chunks


class MyModuleDict(nn.ModuleDict):
    def __init__(self, modules):
        # a mapping (dictionary) of (string: module) or an iterable of key-value pairs of type (string, module)
        if isinstance(modules, dict):
            super().__init__({str(k): v for k, v in modules.items()})
        else:
            raise NotImplementedError

    def __getitem__(self, key):
        return super().__getitem__(str(key))

    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)

    def __delitem__(self, key):
        super().__delitem__(str(key))


class MyHeteroConv(nn.Module):
    """
    Implementation from PyG 2.2.0 with minor changes.
    Override forward pass to control the final aggregation
    Ref.: https://pytorch-geometric.readthedocs.io/en/2.2.0/_modules/torch_geometric/nn/conv/hetero_conv.html
    """
    def __init__(self, convs, aggr="sum"):
        self.vo = {}
        for k, module in convs.items():
            dst = k[-1]
            if dst not in self.vo:
                self.vo[dst] = module.vo
            else:
                assert self.vo[dst] == module.vo

        # from the original implementation in PyTorch Geometric
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behaviour.")

        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.aggr = aggr

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'

    def forward(
            self,
            x_dict,
            edge_index_dict,
            *args_dict,
            **kwargs_dict,
    ):
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict = defaultdict(list)
        out_dict_edge = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            if src == dst:
                out = conv(x_dict[src], edge_index, *args, **kwargs)
            else:
                out = conv((x_dict[src], x_dict[dst]), edge_index, *args,
                           **kwargs)

            if isinstance(out, (tuple, list)):
                out, out_edge = out
                out_dict_edge[edge_type] = out_edge

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)
            out_dict[key] = _split(out_dict[key], self.vo[key])

        return out_dict if len(out_dict_edge) <= 0 else out_dict, out_dict_edge


class GVPHeteroConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    :param update_edge_attr: whether to compute an updated edge representation
    '''

    def __init__(self, in_dims, out_dims, edge_dims, in_dims_other=None,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 update_edge_attr=False):
        super(GVPHeteroConv, self).__init__(aggr=aggr)

        if in_dims_other is None:
            in_dims_other = in_dims

        self.si, self.vi = in_dims
        self.si_other, self.vi_other = in_dims_other
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        self.update_edge_attr = update_edge_attr

        GVP_ = functools.partial(GVP,
                                 activations=activations,
                                 vector_gate=vector_gate)

        def get_modules(module_list, out_dims):
            module_list = module_list or []
            if not module_list:
                if n_layers == 1:
                    module_list.append(
                        GVP_((self.si + self.si_other + self.se, self.vi + self.vi_other + self.ve),
                             (self.so, self.vo), activations=(None, None)))
                else:
                    module_list.append(
                        GVP_((self.si + self.si_other + self.se, self.vi + self.vi_other + self.ve),
                             out_dims)
                    )
                    for i in range(n_layers - 2):
                        module_list.append(GVP_(out_dims, out_dims))
                    module_list.append(GVP_(out_dims, out_dims,
                                            activations=(None, None)))
            return nn.Sequential(*module_list)

        self.message_func = get_modules(module_list, out_dims)
        self.edge_func = get_modules(module_list, edge_dims) if self.update_edge_attr else None

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        elem_0, elem_1 = x
        if isinstance(elem_0, (tuple, list)):
            assert isinstance(elem_1, (tuple, list))
            x_s = (elem_0[0], elem_1[0])
            x_v = (elem_0[1].reshape(elem_0[1].shape[0], 3 * elem_0[1].shape[1]),
                   elem_1[1].reshape(elem_1[1].shape[0], 3 * elem_1[1].shape[1]))
        else:
            x_s, x_v = elem_0, elem_1
            x_v = x_v.reshape(x_v.shape[0], 3 * x_v.shape[1])

        message = self.propagate(edge_index, s=x_s, v=x_v, edge_attr=edge_attr)

        if self.update_edge_attr:
            if isinstance(x_s, (tuple, list)):
                s_i, s_j = x_s[1][edge_index[1]], x_s[0][edge_index[0]]
            else:
                s_i, s_j = x_s[edge_index[1]], x_s[edge_index[0]]

            if isinstance(x_v, (tuple, list)):
                v_i, v_j = x_v[1][edge_index[1]], x_v[0][edge_index[0]]
            else:
                v_i, v_j = x_v[edge_index[1]], x_v[edge_index[0]]

            edge_out = self.edge_attr(s_i, v_i, s_j, v_j, edge_attr)
            # return _split(message, self.vo), edge_out
            return message, edge_out
        else:
            # return _split(message, self.vo)
            return message

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)

    def edge_attr(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        return self.edge_func(message)


class GVPHeteroConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param conv_dims: dictionary defining (src_dim, dst_dim, edge_dim) for each edge type
    """
    def __init__(self, conv_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 update_edge_attr=False, ln_vector_weight=False):

        super(GVPHeteroConvLayer, self).__init__()
        self.update_edge_attr = update_edge_attr

        gvp_conv = partial(GVPHeteroConv,
                           n_layers=n_message,
                           aggr="sum",
                           activations=activations,
                           vector_gate=vector_gate,
                           update_edge_attr=update_edge_attr)

        def get_feedforward(n_dims):
            GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)

            ff_func = []
            if n_feedforward == 1:
                ff_func.append(GVP_(n_dims, n_dims, activations=(None, None)))
            else:
                hid_dims = 4 * n_dims[0], 2 * n_dims[1]
                ff_func.append(GVP_(n_dims, hid_dims))
                for i in range(n_feedforward - 2):
                    ff_func.append(GVP_(hid_dims, hid_dims))
                ff_func.append(GVP_(hid_dims, n_dims, activations=(None, None)))
            return nn.Sequential(*ff_func)

        # self.conv = HeteroConv({k: gvp_conv(*dims) for k, dims in conv_dims.items()}, aggr='sum')
        self.conv = MyHeteroConv({k: gvp_conv(*dims) for k, dims in conv_dims.items()}, aggr='sum')

        node_dims = {k[-1]: dims[1] for k, dims in conv_dims.items()}
        self.norm0 = MyModuleDict({k: gvp.LayerNorm(dims, ln_vector_weight) for k, dims in node_dims.items()})
        self.dropout0 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in node_dims.items()})
        self.ff_func = MyModuleDict({k: get_feedforward(dims) for k, dims in node_dims.items()})
        self.norm1 = MyModuleDict({k: gvp.LayerNorm(dims, ln_vector_weight) for k, dims in node_dims.items()})
        self.dropout1 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in node_dims.items()})

        if self.update_edge_attr:
            self.edge_norm0 = MyModuleDict({k: gvp.LayerNorm(dims[2], ln_vector_weight) for k, dims in conv_dims.items()})
            self.edge_dropout0 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in conv_dims.items()})
            self.edge_ff = MyModuleDict({k: get_feedforward(dims[2]) for k, dims in conv_dims.items()})
            self.edge_norm1 = MyModuleDict({k: gvp.LayerNorm(dims[2], ln_vector_weight) for k, dims in conv_dims.items()})
            self.edge_dropout1 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in conv_dims.items()})

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, node_mask_dict=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''

        dh_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)

        if self.update_edge_attr:
            dh_dict, de_dict = dh_dict

            for k, edge_attr in edge_attr_dict.items():
                de = de_dict[k]

                edge_attr = self.edge_norm0[k](tuple_sum(edge_attr, self.edge_dropout0[k](de)))
                de = self.edge_ff[k](edge_attr)
                edge_attr = self.edge_norm1[k](tuple_sum(edge_attr, self.edge_dropout1[k](de)))

                edge_attr_dict[k] = edge_attr

        for k, x in x_dict.items():
            dh = dh_dict[k]
            node_mask = None if node_mask_dict is None else node_mask_dict[k]

            if node_mask is not None:
                x_ = x
                x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

            x = self.norm0[k](tuple_sum(x, self.dropout0[k](dh)))

            dh = self.ff_func[k](x)
            x = self.norm1[k](tuple_sum(x, self.dropout1[k](dh)))

            if node_mask is not None:
                x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
                x = x_

            x_dict[k] = x

        return (x_dict, edge_attr_dict) if self.update_edge_attr else x_dict


class GVPModel(torch.nn.Module):
    """
    GVP-GNN model
    inspired by: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    and: https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py#L115
    """
    def __init__(self,
                 node_in_dim_ligand,
                 node_in_dim_pocket,
                 edge_in_dim_ligand,
                 edge_in_dim_pocket,
                 edge_in_dim_interaction,
                 node_h_dim_ligand,
                 node_h_dim_pocket,
                 edge_h_dim_ligand,
                 edge_h_dim_pocket,
                 edge_h_dim_interaction,
                 node_out_dim_ligand=None,
                 node_out_dim_pocket=None,
                 edge_out_dim_ligand=None,
                 edge_out_dim_pocket=None,
                 edge_out_dim_interaction=None,
                 num_layers=3,
                 drop_rate=0.1,
                 vector_gate=False,
                 update_edge_attr=False,
                 pocket_encoder_params=None,
                 pocket_node_encoder='gvp',
                 trained_dmasif=None):

        super(GVPModel, self).__init__()

        self.update_edge_attr = update_edge_attr
        self.pocket_node_encoder = pocket_node_encoder

        if self.pocket_node_encoder == 'gvp':
            pocket_encoder = GVP(node_in_dim_pocket, node_h_dim_pocket,
            activations=(None, None), vector_gate=vector_gate)
        elif self.pocket_node_encoder == 'dmasif':
            pocket_encoder = dMaSIF(pocket_encoder_params, return_curvatures=False)
            if trained_dmasif is not None:
                pocket_encoder.load_state_dict(trained_dmasif)
        else:
            raise ValueError(f"Unknown pocket node encoder: {self.pocket_node_encoder}")
        self.node_in = nn.ModuleDict({
            'ligand': GVP(node_in_dim_ligand, node_h_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            'pocket': pocket_encoder,
        })

        self.edge_in = MyModuleDict({
            ('ligand', '', 'ligand'): GVP(edge_in_dim_ligand, edge_h_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            ('pocket', '', 'pocket'): GVP(edge_in_dim_pocket, edge_h_dim_pocket, activations=(None, None), vector_gate=vector_gate),
            ('ligand', '', 'pocket'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
            ('pocket', '', 'ligand'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
        })

        conv_dims = {
            ('ligand', '', 'ligand'): (node_h_dim_ligand, node_h_dim_ligand, edge_h_dim_ligand),
            ('pocket', '', 'pocket'): (node_h_dim_pocket, node_h_dim_pocket, edge_h_dim_pocket),
            ('ligand', '', 'pocket'): (node_h_dim_ligand, node_h_dim_pocket, edge_h_dim_interaction, node_h_dim_pocket),
            ('pocket', '', 'ligand'): (node_h_dim_pocket, node_h_dim_ligand, edge_h_dim_interaction, node_h_dim_ligand),
        }

        self.layers = nn.ModuleList(
            GVPHeteroConvLayer(conv_dims,
                               drop_rate=drop_rate,
                               update_edge_attr=self.update_edge_attr,
                               activations=(F.relu, None),
                               vector_gate=vector_gate,
                               ln_vector_weight=True)
            for _ in range(num_layers))

        self.node_out = nn.ModuleDict({
            'ligand': GVP(node_h_dim_ligand, node_out_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            'pocket': GVP(node_h_dim_pocket, node_out_dim_pocket, activations=(None, None), vector_gate=vector_gate) if node_out_dim_pocket is not None else None,
        })

        self.edge_out = MyModuleDict({
            ('ligand', '', 'ligand'): GVP(edge_h_dim_ligand, edge_out_dim_ligand, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_ligand is not None else None,
            ('pocket', '', 'pocket'): GVP(edge_h_dim_pocket, edge_out_dim_pocket, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_pocket is not None else None,
            ('ligand', '', 'pocket'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
            ('pocket', '', 'ligand'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
        })

    def forward(self, node_attr, batch_mask, edge_index, edge_attr):

        # to hidden dimension
        for k in node_attr.keys():
            node_attr[k] = self.node_in[k](node_attr[k])

        for k in edge_attr.keys():
            edge_attr[k] = self.edge_in[k](edge_attr[k])

        # convolutions
        for layer in self.layers:
            out = layer(node_attr, edge_index, edge_attr)
            if self.update_edge_attr:
                node_attr, edge_attr = out
            else:
                node_attr = out

        # to output dimension
        for k in node_attr.keys():
            if self.node_out[k] is not None:
                node_attr[k] = self.node_out[k](node_attr[k])

        if self.update_edge_attr:
            for k in edge_attr.keys():
                if self.edge_out[k] is not None:
                    edge_attr[k] = self.edge_out[k](edge_attr[k])

        return node_attr, edge_attr


class DynamicsHetero(DynamicsBase):
    def __init__(self,
                 atom_nf,
                 residue_nf,
                 bond_dict,
                 pocket_bond_nf,
                 condition_time=True,
                 model='gvp',
                 pocket_node_encoder='gvp',
                 model_params=None,
                 edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None,
                 edge_cutoff_interaction=None,
                 add_spectral_feat=False,
                 add_nma_feat=False,
                 reflection_equiv=False,
                 d_max=15.0,
                 num_rbf=16,
                 self_conditioning=False,
                 add_all_atom_diff=False,
                 pocket_encoder_params=None,
                 k_neighbours=10,
                 predict_frames=False,
                 predict_edges=True,):

        super().__init__(
            predict_frames=predict_frames,
            add_spectral_feat=add_spectral_feat,
            self_conditioning=self_conditioning,
            predict_edges=predict_edges,
        )

        self.model = model
        self.atom_nf = atom_nf
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.bond_dict = bond_dict
        if predict_edges:
            self.bond_nf = len(bond_dict)
        else:
            self.bond_nf = 0
        self.pocket_bond_nf = pocket_bond_nf
        self.add_spectral_feat = add_spectral_feat
        self.add_nma_feat = add_nma_feat
        self.self_conditioning = self_conditioning
        self.add_all_atom_diff = add_all_atom_diff
        self.pocket_node_encoder = pocket_node_encoder
        self.k_neighbours = k_neighbours
        self.predict_frames = predict_frames
        self.predict_edges = predict_edges

        # edge encoding params
        self.reflection_equiv = reflection_equiv
        self.d_max = d_max
        self.num_rbf = num_rbf

        if self.pocket_node_encoder == 'dmasif':
            if 'trained_dmasif' in pocket_encoder_params:
                trained_dmasif = torch.load(pocket_encoder_params.trained_dmasif, map_location='cpu')
                trained_dmasif_trim = {k:v for k, v in trained_dmasif['state_dict'].items() \
                                       if not k.startswith('ligand_encoder') and not k.startswith('nci_classifier')}
                trained_dmasif_trim = {k.replace('surface_encoder.', ''):v for k, v in trained_dmasif_trim.items()}
                dmasif_params = trained_dmasif['hyper_parameters']['surface_encoder_params']
                # namespace
                dmasif_params = argparse.Namespace(**dmasif_params)
            else:
                dmasif_params = pocket_encoder_params
                trained_dmasif_trim = None

            pocket_node_h_dim = (dmasif_params.emb_dims, 0)
            if self.add_nma_feat:
                pocket_node_h_dim = (pocket_node_h_dim[0], 1)
        else:
            pocket_node_h_dim = model_params.node_h_dim
            dmasif_params = None
            trained_dmasif_trim = None

        # Output dimensions dimensions, always tuple (scalar, vector)
        _atom_out = (atom_nf[0], 1) if isinstance(atom_nf, Iterable) else (atom_nf, 1)

        if self.predict_frames:
            # unlike for residues do only predict rotation as x is separately (otherwise add (3, 1))
            _atom_out = tuple_sum(_atom_out, (3, 0))

        # Input dimensions dimensions, always tuple (scalar, vector)
        _atom_in = (atom_nf, 0)
        _residue_in = tuple(residue_nf)
        _residue_atom_dim = residue_nf[1]

        if self.add_spectral_feat:
            _atom_in = tuple_sum(_atom_in, (5, 0))

        if self.add_nma_feat:
            if self.pocket_node_encoder == 'dmasif':
                _residue_in = tuple_sum(_residue_in, (0, 1))
            else:
                _residue_in = tuple_sum(_residue_in, (0, 5))

        if self.self_conditioning:
            _atom_in = tuple_sum(_atom_in, _atom_out)

        if self.predict_frames:
            _atom_in = tuple_sum(_atom_in, (3, 0))

        self.condition_time = condition_time
        if self.condition_time:
            _atom_in = tuple_sum(_atom_in, (1, 0))
            if self.pocket_node_encoder != 'dmasif':
                _residue_in = tuple_sum(_residue_in, (1, 0))
        else:
            print('Warning: dynamics model is NOT conditioned on time.')

        # Edge output dimensions, always tuple (scalar, vector)
        _edge_ligand_out = (self.bond_nf, 0)
        _edge_ligand_before_symmetrization = (model_params.edge_h_dim[0], 0) if self.predict_edges else None

        # Edge input dimensions dimensions, always tuple (scalar, vector)
        _edge_ligand_in = (self.bond_nf + self.num_rbf, 1 if self.reflection_equiv else 2)
        _edge_ligand_in = tuple_sum(_edge_ligand_in, _atom_in)  # src node
        _edge_ligand_in = tuple_sum(_edge_ligand_in, _atom_in)  # dst node

        if self_conditioning and self.predict_edges:
            _edge_ligand_in = tuple_sum(_edge_ligand_in, _edge_ligand_out)

        _n_dist_residue = _residue_atom_dim ** 2 if self.add_all_atom_diff else 1
        _edge_pocket_in = (_n_dist_residue * self.num_rbf + self.pocket_bond_nf, _n_dist_residue)
        _edge_pocket_in = tuple_sum(_edge_pocket_in, _residue_in)  # src node
        _edge_pocket_in = tuple_sum(_edge_pocket_in, _residue_in)  # dst node

        _n_dist_interaction = _residue_atom_dim if self.add_all_atom_diff else 1
        _edge_interaction_in = (_n_dist_interaction * self.num_rbf, _n_dist_interaction)
        _edge_interaction_in = tuple_sum(_edge_interaction_in, _atom_in)  # atom node
        _edge_interaction_in = tuple_sum(_edge_interaction_in, _residue_in)  # residue node

        # Embeddings for newly added edges
        if self.predict_edges:
            _ligand_nobond_nf = self.bond_nf + _edge_ligand_out[0] if self.self_conditioning else self.bond_nf
            self.ligand_nobond_emb = nn.Parameter(torch.zeros(_ligand_nobond_nf), requires_grad=True)
            if self.pocket_bond_nf > 0:
                self.pocket_nobond_emb = nn.Parameter(torch.zeros(self.pocket_bond_nf), requires_grad=True)

        # for access in self-conditioning
        self.atom_out_dim = _atom_out
        self.edge_out_dim = _edge_ligand_out

        if model == 'gvp':
            self.net = GVPModel(
                node_in_dim_ligand=_atom_in,
                node_in_dim_pocket=_residue_in,
                edge_in_dim_ligand=_edge_ligand_in,
                edge_in_dim_pocket=_edge_pocket_in,
                edge_in_dim_interaction=_edge_interaction_in,
                node_h_dim_ligand=model_params.node_h_dim,
                node_h_dim_pocket=pocket_node_h_dim,
                edge_h_dim_ligand=model_params.edge_h_dim,
                edge_h_dim_pocket=model_params.edge_h_dim,
                edge_h_dim_interaction=model_params.edge_h_dim,
                node_out_dim_ligand=_atom_out,
                node_out_dim_pocket=None,
                edge_out_dim_ligand=_edge_ligand_before_symmetrization,
                edge_out_dim_pocket=None,
                edge_out_dim_interaction=None,
                num_layers=model_params.n_layers,
                drop_rate=model_params.dropout,
                vector_gate=model_params.vector_gate,
                update_edge_attr=True,
                pocket_encoder_params=dmasif_params,
                pocket_node_encoder=self.pocket_node_encoder,
                trained_dmasif=trained_dmasif_trim,
            )

        else:
            raise NotImplementedError(f"{model} is not available")

        assert _edge_ligand_out[1] == 0

        if self.predict_edges:
            assert _edge_ligand_before_symmetrization[1] == 0
            self.edge_decoder = nn.Sequential(
                nn.Linear(_edge_ligand_before_symmetrization[0], _edge_ligand_before_symmetrization[0]),
                torch.nn.SiLU(),
                nn.Linear(_edge_ligand_before_symmetrization[0], _edge_ligand_out[0])
            )

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
        if self.pocket_node_encoder == 'dmasif':
            x_residues, h_residues, mask_residues = pocket['x_surface'], None, pocket['mask_surface']
        else:
            x_residues, h_residues, mask_residues = pocket['x'], pocket['one_hot'], pocket['mask']
        if 'bonds' in pocket and self.pocket_node_encoder != 'dmasif':
            bonds_pocket = (pocket['bonds'], pocket['bond_one_hot'])
        else:
            bonds_pocket = None

        if self.pocket_node_encoder == 'dmasif' and self.add_nma_feat:
            vecs = pocket['normals_surface']
        elif self.pocket_node_encoder != 'dmasif' and 'v' in pocket:
            vecs = pocket['v']
            v_residues = pocket['v']
            if self.add_nma_feat:
                v_residues = torch.cat([v_residues, pocket['nma_vec']], dim=1)
            h_residues = (h_residues, v_residues)
        else:
            vecs = None

        # NOTE: 'bond' denotes one-directional edges and 'edge' means bi-directional
        # get graph edges and edge attributes
        if bonds_ligand is not None:

            ligand_bond_indices = bonds_ligand[0]

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
        else:
            pocket_edge_indices = None
            pocket_edge_types = None


        # Self-conditioning
        if h_atoms_sc is not None:
            h_atoms = (torch.cat([h_atoms, h_atoms_sc[0]], dim=-1),
                       h_atoms_sc[1])

        if e_atoms_sc is not None:
            e_atoms_sc = torch.cat([e_atoms_sc, e_atoms_sc], dim=0)
            ligand_edge_types = torch.cat([ligand_edge_types, e_atoms_sc], dim=-1)

        if h_residues_sc is not None:
            h_residues = (torch.cat([h_residues[0], h_residues_sc], dim=-1),
                            h_residues[1])

        if self.condition_time:
            h_atoms = (torch.cat([h_atoms[0], t[mask_atoms]], dim=1), h_atoms[1])
            if h_residues is not None:
                h_residues = (torch.cat([h_residues[0], t[mask_residues]], dim=1), h_residues[1])

        # Process edges and encode in shared feature space
        edge_index_dict, edge_attr_dict = self.get_edges(
            x_atoms, h_atoms, mask_atoms, ligand_edge_indices, ligand_edge_types,
            x_residues, h_residues, mask_residues, vecs, pocket_edge_indices, pocket_edge_types)

        if self.pocket_node_encoder == 'dmasif':
            pocket = {k: pocket[k] for k in ['x_surface', 'normals_surface',
                                             'mask_surface', 'atom_xyz_surface',
                                             'atomtypes_surface', 'atom_batch_surface']}
            # rename keys to match the model
            pocket['xyz'] = pocket.pop('x_surface')
            pocket['normals'] = pocket.pop('normals_surface')
            pocket['batch'] = pocket.pop('mask_surface')
            pocket['atom_xyz'] = pocket.pop('atom_xyz_surface')
            pocket['atomtypes'] = pocket.pop('atomtypes_surface')
            pocket['batch_atoms'] = pocket.pop('atom_batch_surface')
            node_attr_dict = {
                'ligand': h_atoms,
                'pocket': pocket,
            }
        elif self.pocket_node_encoder == 'gvp':
            node_attr_dict = {
                'ligand': h_atoms,
                'pocket': h_residues,
            }
        else:
            raise ValueError(f"Unknown pocket node encoder: {self.pocket_node_encoder}")
        batch_mask_dict = {
            'ligand': mask_atoms,
            'pocket': mask_residues,
        }

        if self.model == 'gvp':
            out_node_attr, out_edge_attr = self.net(
                node_attr_dict, batch_mask_dict, edge_index_dict, edge_attr_dict)
        else:
            raise NotImplementedError(f"Wrong model ({self.model})")

        h_final_atoms = out_node_attr['ligand'][0][:, :self.atom_nf].clone()
        vel = out_node_attr['ligand'][1].squeeze(-2)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final_atoms)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                h_final_atoms[torch.isnan(h_final_atoms)] = 0.0
            else:
                raise ValueError("NaN detected in network output")

        pred_ligand = {
            'vel': vel,
            'logits_h': h_final_atoms,
        }

        if self.predict_edges:
            edge_final = out_edge_attr[('ligand', '', 'ligand')]
            edges = edge_index_dict[('ligand', '', 'ligand')]

            # Symmetrize
            edge_logits = torch.zeros(
                (len(mask_atoms), len(mask_atoms), edge_final.size(-1)),
                device=mask_atoms.device)
            edge_logits[edges[0], edges[1]] = edge_final
            edge_logits = (edge_logits + edge_logits.transpose(0, 1)) * 0.5

            # return upper triangular elements only (matching the input)
            edge_logits = edge_logits[ligand_bond_indices[0], ligand_bond_indices[1]]

            edge_final_atoms = self.edge_decoder(edge_logits)
            pred_ligand['logits_e'] = edge_final_atoms

        if self.predict_frames:
            pred_ligand['rot_vec'] = out_node_attr['ligand'][0][:, self.atom_nf:].clone()

        return pred_ligand

    def get_edges(self, x_ligand, h_ligand, batch_mask_ligand, edges_ligand, edge_feat_ligand,
                  x_pocket, h_pocket, batch_mask_pocket, atom_vectors_pocket, edges_pocket, edge_feat_pocket,
                  self_edges=False):

        # Adjacency matrix
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

            # Add missing bonds if they got removed
            adj_ligand[edges_ligand[0], edges_ligand[1]] = True

        if self.edge_cutoff_p is not None:
            if self.pocket_node_encoder == 'dmasif':
                indices, _ = knn_atoms(x_pocket, x_pocket, batch_mask_pocket, batch_mask_pocket, k=self.k_neighbours)
                indices = indices.reshape(-1, self.k_neighbours)
                n_atoms = indices.shape[0]
                i_tensor = torch.arange(n_atoms, device=indices.device).unsqueeze(1).expand_as(indices)

                # Create the adjacency matrix in one operation
                adj_pocket = torch.zeros(n_atoms, n_atoms, dtype=torch.bool, device=indices.device)
                adj_pocket[i_tensor, indices] = True
                adj_pocket = adj_pocket & adj_pocket.T
            else:
                adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

            # Add missing bonds if they got removed
            if edges_pocket is not None:
                adj_pocket[edges_pocket[0], edges_pocket[1]] = True

        if not self_edges:
            adj_ligand = adj_ligand ^ torch.eye(*adj_ligand.size(), out=torch.empty_like(adj_ligand))
            # add self-edges to ligands with only one node
            adj_ligand = self.add_single_self_edges(adj_ligand, batch_mask_ligand)
            # adj_pocket = adj_pocket ^ torch.eye(*adj_pocket.size(), out=torch.empty_like(adj_pocket))
            torch.diagonal(adj_pocket)[:] = 0

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)
            # skip batch if no cross edges
            if not adj_cross.any():
                raise RuntimeError("No cross edges found - skip batch")

        # ligand-ligand edge features
        edges_ligand_updated = torch.stack(torch.where(adj_ligand), dim=0)
        if self.bond_nf > 0:
            feat_ligand = self.ligand_nobond_emb.repeat(*adj_ligand.shape, 1)
            feat_ligand[edges_ligand[0], edges_ligand[1]] = edge_feat_ligand
            feat_ligand = feat_ligand[edges_ligand_updated[0], edges_ligand_updated[1]]
        else:
            feat_ligand = None
        feat_ligand = self.ligand_edge_features(h_ligand, x_ligand, edges_ligand_updated, batch_mask_ligand, edge_attr=feat_ligand)

        # residue-residue edge features
        edges_pocket_updated = process_in_chunks(adj_pocket)  # with surface rep can exceed mem limits
        if self.pocket_bond_nf > 0:
            feat_pocket = self.pocket_nobond_emb.repeat(*adj_pocket.shape, 1)
            feat_pocket[edges_pocket[0], edges_pocket[1]] = edge_feat_pocket
            feat_pocket = feat_pocket[edges_pocket_updated[0], edges_pocket_updated[1]]
        else:
            feat_pocket = None
        feat_pocket = self.pocket_edge_features(h_pocket, x_pocket, atom_vectors_pocket, edges_pocket_updated, edge_attr=feat_pocket)

        # ligand-residue edge features
        edges_cross = torch.stack(torch.where(adj_cross), dim=0)
        feat_cross = self.cross_edge_features(h_ligand, x_ligand, h_pocket, x_pocket, atom_vectors_pocket, edges_cross)

        edge_index = {
            ('ligand', '', 'ligand'): edges_ligand_updated,
            ('pocket', '', 'pocket'): edges_pocket_updated,
            ('ligand', '', 'pocket'): edges_cross,
            ('pocket', '', 'ligand'): edges_cross.flip(dims=[0]),
        }

        edge_attr = {
            ('ligand', '', 'ligand'): feat_ligand,
            ('pocket', '', 'pocket'): feat_pocket,
            ('ligand', '', 'pocket'): feat_cross,
            ('pocket', '', 'ligand'): feat_cross,
        }

        return edge_index, edge_attr

    def add_single_self_edges(self, adj, batch_mask):
        # Create a device-compatible eye tensor
        device = adj.device

        # Create a mask for adding self-edges
        self_edge_mask = torch.zeros_like(adj, dtype=torch.bool)

        # Iterate through unique samples in the batch
        for sample_idx in range(batch_mask.max().item() + 1):
            # Find the range of atoms for this sample
            sample_atoms = (batch_mask == sample_idx)

            if sum(sample_atoms) == 1:
                # Add a self-edge if there is only one atom in the sample
                self_edge_mask[sample_atoms, sample_atoms] = True

        # Add self-edges using the mask
        adj = adj | self_edge_mask.to(device)

        return adj

    def ligand_edge_features(self, h, x, edge_index, batch_mask=None, edge_attr=None):
        """
        :param h: (s, V)
        :param x:
        :param edge_index:
        :param batch_mask:
        :param edge_attr:
        :return: scalar and vector-valued edge features
        """
        row, col = edge_index
        coord_diff = x[row] - x[col]
        dist = coord_diff.norm(dim=-1)
        rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf,
                   device=x.device)

        edge_s = torch.cat([h[0][row], h[0][col], rbf], dim=1)
        edge_v = torch.cat([h[1][row], h[1][col], _normalize(coord_diff).unsqueeze(-2)], dim=1)

        if edge_attr is not None:
            edge_s = torch.cat([edge_s, edge_attr], dim=1)

        # self.reflection_equiv: bool, use reflection-sensitive feature based on
        #                        the cross product if False
        if not self.reflection_equiv:
            mean = scatter_mean(x, batch_mask, dim=0,
                                dim_size=batch_mask.max() + 1)
            row, col = edge_index
            cross = torch.cross(x[row] - mean[batch_mask[row]],
                                x[col] - mean[batch_mask[col]], dim=1)
            cross = _normalize(cross).unsqueeze(-2)

            edge_v = torch.cat([edge_v, cross], dim=-2)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)

    def pocket_edge_features(self, h, x, v, edge_index, edge_attr=None):
        """
        :param h: (s, V)
        :param x:
        :param v:
        :param edge_index:
        :param edge_attr:
        :return: scalar and vector-valued edge features
        """
        row, col = edge_index

        if self.add_all_atom_diff:
            all_coord = v + x.unsqueeze(1)  # (nR, nA, 3)
            coord_diff = all_coord[row, :, None, :] - all_coord[col, None, :, :]  # (nB, nA, nA, 3)
            coord_diff = coord_diff.flatten(1, 2)
            dist = coord_diff.norm(dim=-1)  # (nB, nA^2)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x.device)  # (nB, nA^2, rdb_dim)
            rbf = rbf.flatten(1, 2)
            coord_diff = _normalize(coord_diff)
        else:
            coord_diff = x[row] - x[col]
            dist = coord_diff.norm(dim=-1)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x.device)
            coord_diff = _normalize(coord_diff).unsqueeze(-2)

        if h is not None:
            edge_s = torch.cat([h[0][row], h[0][col], rbf], dim=1)
            edge_v = torch.cat([h[1][row], h[1][col], coord_diff], dim=1)
        elif self.add_nma_feat:
            edge_s = rbf
            v = v.unsqueeze(1)
            edge_v = torch.cat([v[row], v[col], coord_diff], dim=1)
        else:
            edge_s = rbf
            edge_v = coord_diff

        if edge_attr is not None:
            edge_s = torch.cat([edge_s, edge_attr], dim=1)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)

    def cross_edge_features(self, h_ligand, x_ligand, h_pocket, x_pocket, v_pocket, edge_index):
        """
        :param h_ligand: (s, V)
        :param x_ligand:
        :param h_pocket: (s, V)
        :param x_pocket:
        :param v_pocket:
        :param edge_index: first row indexes into the ligand tensors, second row into the pocket tensors

        :return: scalar and vector-valued edge features
        """
        ligand_idx, pocket_idx = edge_index

        if self.add_all_atom_diff:
            all_coord_pocket = v_pocket + x_pocket.unsqueeze(1)  # (nR, nA, 3)
            coord_diff = x_ligand[ligand_idx, None, :] - all_coord_pocket[pocket_idx]  # (nB, nA, 3)
            dist = coord_diff.norm(dim=-1)  # (nB, nA)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x_ligand.device)  # (nB, nA, rdb_dim)
            rbf = rbf.flatten(1, 2)
            coord_diff = _normalize(coord_diff)
        else:
            coord_diff = x_ligand[ligand_idx] - x_pocket[pocket_idx]
            dist = coord_diff.norm(dim=-1)  # (nB, nA)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x_ligand.device)
            coord_diff = _normalize(coord_diff).unsqueeze(-2)

        if h_pocket is not None:
            edge_s = torch.cat([h_ligand[0][ligand_idx], h_pocket[0][pocket_idx], rbf], dim=1)
            edge_v = torch.cat([h_ligand[1][ligand_idx], h_pocket[1][pocket_idx], coord_diff], dim=1)
        elif self.add_nma_feat:
            edge_s = torch.cat([h_ligand[0][ligand_idx], rbf], dim=1)
            v_pocket = v_pocket.unsqueeze(1)
            edge_v = torch.cat([h_ligand[1][ligand_idx], v_pocket[pocket_idx], coord_diff], dim=1)
        else:
            edge_s = torch.cat([h_ligand[0][ligand_idx], rbf], dim=1)
            edge_v = torch.cat([h_ligand[1][ligand_idx], coord_diff], dim=1)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)
