import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter as t_scatter

from latentfrag.encoder.utils.scatter import scatter
from latentfrag.encoder.ligand_encoder.embedding import RadialBesselBasis
from latentfrag.encoder.utils.constants import bond_mapping


class GCL(nn.Module):
    def __init__(self, node_nf, hidden_nf, edge_nf=0, aggregation_method='sum',
                 act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()

        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_nf + edge_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            # act_fn
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_nf + hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, node_nf)
        )

        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid()
        ) if self.attention else None

    def forward(self, h, edge_index, edge_attr=None):
        row, col = edge_index
        m_ij = self.message(h[row], h[col], edge_attr)

        dh = self.aggregate(h, edge_index, m_ij)
        h = h + dh

        return h

    def message(self, source, target, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)

        mij = self.edge_mlp(out)

        if self.attention:
            mij = mij * self.att_mlp(mij)

        return mij

    def aggregate(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = scatter(edge_attr, index=row, dim=0, dim_size=h.size(0),
                      reduce=self.aggregation_method)

        agg = torch.cat([h, agg], dim=1)
        return self.node_mlp(agg)


class GNN(nn.Module):
    def __init__(self, in_node_nf, out_node_nf, hidden_nf, in_edge_nf, n_layers,
                 aggregation_method='sum', act_fn=nn.SiLU(), attention=False, custom_init=False,):
        super(GNN, self).__init__()

        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.embedding_in = nn.Sequential(
            nn.Linear(self.in_node_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf)
        )

        for i in range(n_layers):
            self.add_module(f"norm_{i}", nn.LayerNorm(self.hidden_nf))
            self.add_module(
                f"gcl_{i}", GCL(self.hidden_nf, self.hidden_nf, in_edge_nf,
                                aggregation_method=aggregation_method,
                                act_fn=act_fn, attention=attention))

        self.embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.out_node_nf),
        )

        if custom_init == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

        elif custom_init == 'uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0.0)

        elif custom_init == 'constant':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0.0)
                    nn.init.constant_(m.bias, 0.0)

        elif custom_init == 'normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.1)
                    nn.init.constant_(m.bias, 0.0)


    def forward(self, h, edges, edge_attr=None):

        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h = self._modules[f"norm_{i}"](h)
            h = self._modules[f"gcl_{i}"](h, edges, edge_attr=edge_attr)
        h = self.embedding_out(h)

        return h


class InvariantGraphEncoder(nn.Module):
    def __init__(self,
                 input_node_nf,
                 hidden_node_nf,
                 output_node_nf,
                 input_edge_nf,
                 n_layers,
                 distance_embedding_nf,
                 edge_cutoff,
                 aggregation_method='sum',
                 attention=False,
                 use_distances=False,
                 custom_init=False,
                 use_ev=False,):
        super(InvariantGraphEncoder, self).__init__()

        self.edge_cutoff = edge_cutoff
        self.use_distances = use_distances
        self.input_edge_nf = input_edge_nf
        if use_distances:
            emb_cutoff = 16.0 if self.edge_cutoff is None else self.edge_cutoff
            self.input_edge_nf += distance_embedding_nf
            self.distance_embedding = RadialBesselBasis(
                cutoff=emb_cutoff, num_radial=distance_embedding_nf)

        if use_ev:
            # 15 chemical environments of BRICS (no double bonds) + 1 hybridization state
            input_node_nf += (15 + 1)

        self.gnn = GNN(
            in_node_nf=input_node_nf,
            out_node_nf=output_node_nf,
            in_edge_nf=self.input_edge_nf,
            hidden_nf=hidden_node_nf,
            n_layers=n_layers,
            aggregation_method=aggregation_method,
            attention=attention,
            custom_init=custom_init,
        )

    def forward(self,
                x,
                h,
                batch_mask,
                covalent_bonds,
                bond_types,
                mol_feats,
                return_global=False):
        """
        :param x: (n, 3)
        :param h: (n, nf)
        :param batch_mask: (n,)
        :param covalent_bonds: (2, n_bonds)
        :param bond_types: (n_bonds, one_hot_dim)
        :param mol_feats: (n_mol_feats,)
        :param return_global: bool
        :return: (n, nf_out)
        """

        edges, edge_feat = self.get_edges(batch_mask, x, covalent_bonds,
                                          bond_types)

        if self.use_distances:
            distances = self.coord2diff(x, edges)
            distances = self.distance_embedding(distances)
            edge_feat = torch.cat([edge_feat, distances], dim=1)

        h = self.gnn(h.float(), edges, edge_attr=edge_feat.float())

        if return_global:
            h_global = t_scatter(h, batch_mask, reduce='mean', dim=0)
            return h, h_global
        else:
            return h

    def coord2diff(self, x, edge_index):
        row, col = edge_index
        radial = torch.sum((x[row] - x[col]) ** 2, dim=1).sqrt()
        return radial

    def get_edges(self, batch_mask, x, covalent_bonds, bond_types,
                  self_edges=False):

        adj = batch_mask[:, None] == batch_mask[None, :]

        if not self_edges:
            adj = adj ^ torch.eye(*adj.size(), out=torch.empty_like(adj))

        if self.edge_cutoff is not None:
            adj = adj & (torch.cdist(x, x) <= self.edge_cutoff)

        # remove covalent bonds
        row, col = covalent_bonds
        # adj_covalent = torch.zeros_like(adj)
        # adj_covalent[row, col] = 1
        # adj = adj ^ adj_covalent
        adj[row, col] = 0

        edges = torch.stack(torch.where(adj), dim=0)

        # define edge features
        edge_feat = torch.ones(edges.size(1), device=x.device, dtype=int) * bond_mapping["NOBOND"]
        edge_feat = F.one_hot(edge_feat, num_classes=len(bond_mapping))

        # insert covalent bonds again
        edges = torch.cat([edges, covalent_bonds], dim=1)
        edge_feat = torch.cat([edge_feat, bond_types], dim=0)

        return edges, edge_feat
