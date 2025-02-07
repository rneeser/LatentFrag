import torch
import torch.nn as nn
import torch.nn.functional as F
from pykeops.torch import LazyTensor

from latentfrag.encoder.dmasif.geometry_processing import (
    curvatures,
    mesh_normals_areas,
    tangent_vectors,
    dMaSIFConv
)
from latentfrag.encoder.dmasif.helper import diagonal_ranges


def knn_atoms(x, y, x_batch, y_batch, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def get_atom_features(x, y, x_batch, y_batch, y_atomtype, k=16):

    idx, dists = knn_atoms(x, y, x_batch, y_batch, k=k)  # (num_points, k)
    num_points, _ = idx.size()

    idx = idx.view(-1)
    dists = 1 / dists.view(-1, 1)
    _, num_dims = y_atomtype.size()

    feature = y_atomtype[idx, :]
    feature = torch.cat([feature, dists], dim=1)
    feature = feature.view(num_points, k, num_dims + 1)

    return feature


# class Atom_embedding(nn.Module):
#     def __init__(self, args):
#         super(Atom_embedding, self).__init__()
#         self.D = args.atom_dims
#         self.k = 16
#         self.conv1 = nn.Linear(self.D + 1, self.D)
#         self.conv2 = nn.Linear(self.D, self.D)
#         self.conv3 = nn.Linear(2 * self.D, self.D)
#         self.bn1 = nn.BatchNorm1d(self.D)
#         self.bn2 = nn.BatchNorm1d(self.D)
#         self.relu = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, x, y, y_atomtypes, x_batch, y_batch):
#         fx = get_atom_features(x, y, x_batch, y_batch, y_atomtypes, k=self.k)
#         fx = self.conv1(fx)
#         fx = fx.view(-1, self.D)
#         fx = self.bn1(self.relu(fx))
#         fx = fx.view(-1, self.k, self.D)
#         fx1 = fx.sum(dim=1, keepdim=False)
#
#         fx = self.conv2(fx)
#         fx = fx.view(-1, self.D)
#         fx = self.bn2(self.relu(fx))
#         fx = fx.view(-1, self.k, self.D)
#         fx2 = fx.sum(dim=1, keepdim=False)
#         fx = torch.cat((fx1, fx2), dim=-1)
#         fx = self.conv3(fx)
#
#         return fx


# class AtomNet(nn.Module):
#     def __init__(self, args):
#         super(AtomNet, self).__init__()
#         self.args = args
#
#         self.transform_types = nn.Sequential(
#             nn.Linear(args.atom_dims, args.atom_dims),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(args.atom_dims, args.atom_dims),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(args.atom_dims, args.atom_dims),
#             nn.LeakyReLU(negative_slope=0.2),
#         )
#         self.embed = Atom_embedding(args)
#
#     def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
#         # Run a DGCNN on the available information:
#         atomtypes = self.transform_types(atomtypes)
#         return self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)


class Atom_embedding_MP(nn.Module):
    def __init__(self, atom_dims):
        super(Atom_embedding_MP, self).__init__()
        self.D = atom_dims
        self.k = 16
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomtypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atomtypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb


class Atom_Atom_embedding_MP(nn.Module):
    def __init__(self, atom_dims):
        super(Atom_Atom_embedding_MP, self).__init__()
        self.D = atom_dims
        self.k = 17
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomtypes.shape[0]

        out = y_atomtypes
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out


class AtomNet_MP(nn.Module):
    def __init__(self, atom_dims):
        super(AtomNet_MP, self).__init__()

        self.transform_types = nn.Sequential(
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(atom_dims, atom_dims),
        )

        self.embed = Atom_embedding_MP(atom_dims)
        self.atom_atom = Atom_Atom_embedding_MP(atom_dims)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        atomtypes = self.atom_atom(
            atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        )
        atomtypes = self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)
        return atomtypes


class dMaSIFConv_seg(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, radius=9.0):
        super(dMaSIFConv_seg, self).__init__()

        self.name = "dMaSIFConv_seg_keops"
        self.radius = radius
        self.I, self.H, self.O = in_channels, hidden_channels, out_channels

        if n_layers > 2:
            self.layers = nn.ModuleList(
                [dMaSIFConv(self.I, self.H, radius, self.H)] +
                [dMaSIFConv(self.H, self.H, radius, self.H) for _ in range(n_layers - 2)] +
                [dMaSIFConv(self.H, self.O, radius, self.H)]
            )
            self.linear_transform = nn.ModuleList(
                [nn.Linear(self.I, self.H)] +
                [nn.Linear(self.H, self.H) for _ in range(n_layers - 2)] +
                [nn.Linear(self.H, self.O)]
            )
        else:
            self.layers = nn.ModuleList([
                dMaSIFConv(self.I, self.H, radius, self.H),
                dMaSIFConv(self.H, self.O, radius, self.H),
            ])
            self.linear_transform = nn.ModuleList([
                nn.Linear(self.I, self.H),
                nn.Linear(self.H, self.O),
            ])

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.H, self.H), nn.ReLU(), nn.Linear(self.H, self.H))
                for _ in range(n_layers - 1)
            ] +
            [nn.Sequential(nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O))]
        )

    def forward(self, features):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        points, nuv, ranges = self.points, self.nuv, self.ranges
        x = features
        for i, layer in enumerate(self.layers):

            # with profiler.record_function(f"LAYER dMaSIFConv-{i}"):
            # CPU time avg: 3.5s
            x_i = layer(points, nuv, x, ranges)

            # with profiler.record_function(f"LAYER  Linear-Layer-{i}"):
            # CPU time avg: 10-500us
            x_i = self.linear_layers[i](x_i)

            # with profiler.record_function(f"LAYER  Linear-Transform-{i}"):
            # CPU time avg: 500us-1.5ms
            x = self.linear_transform[i](x)
            x = x + x_i

        return x

    def load_mesh(self, xyz, triangles=None, normals=None, weights=None, batch=None):
        """Loads the geometry of a triangle mesh.

        Input arguments:
        - xyz, a point cloud encoded as an (N, 3) Tensor.
        - triangles, a connectivity matrix encoded as an (N, 3) integer tensor.
        - weights, importance weights for the orientation estimation, encoded as an (N, 1) Tensor.
        - radius, the scale used to estimate the local normals.
        - a batch vector, following PyTorch_Geometric's conventions.

        The routine updates the model attributes:
        - points, i.e. the point cloud itself,
        - nuv, a local oriented basis in R^3 for every point,
        - ranges, custom KeOps syntax to implement batch processing.
        """

        # 1. Save the vertices for later use in the convolutions ---------------
        self.points = xyz
        self.batch = batch
        self.ranges = diagonal_ranges(
            batch
        )  # KeOps support for heterogeneous batch processing
        self.triangles = triangles
        self.normals = normals
        self.weights = weights

        # 2. Estimate the normals and tangent frame ----------------------------
        # Normalize the scale:
        points = xyz / self.radius

        # Normals and local areas:
        if normals is None:
            normals, areas = mesh_normals_areas(points, triangles, 0.5, batch)
        tangent_bases = tangent_vectors(normals)  # Tangent basis (N, 2, 3)

        # 3. Steer the tangent bases according to the gradient of "weights" ----

        # 3.a) Encoding as KeOps LazyTensors:
        # Orientation scores:
        weights_j = LazyTensor(weights.view(1, -1, 1))  # (1, N, 1)
        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)
        # Normals:
        n_i = LazyTensor(normals[:, None, :])  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)
        # Tangent basis:
        uv_i = LazyTensor(tangent_bases.view(-1, 1, 6))  # (N, 1, 6)

        # 3.b) Pseudo-geodesic window:
        # Pseudo-geodesic squared distance:
        rho2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
        # Gaussian window:
        window_ij = (-rho2_ij).exp()  # (N, N, 1)

        # 3.c) Coordinates in the (u, v) basis - not oriented yet:
        X_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)

        # 3.d) Local average in the tangent plane:
        orientation_weight_ij = window_ij * weights_j  # (N, N, 1)
        orientation_vector_ij = orientation_weight_ij * X_ij  # (N, N, 2)

        # Support for heterogeneous batch processing:
        orientation_vector_ij.ranges = self.ranges  # Block-diagonal sparsity mask

        orientation_vector_i = orientation_vector_ij.sum(dim=1)  # (N, 2)
        orientation_vector_i = (
            orientation_vector_i + 1e-5
        )  # Just in case someone's alone...

        # 3.e) Normalize stuff:
        orientation_vector_i = F.normalize(orientation_vector_i, p=2, dim=-1)  #  (N, 2)
        ex_i, ey_i = (
            orientation_vector_i[:, 0][:, None],
            orientation_vector_i[:, 1][:, None],
        )  # (N,1)

        # 3.f) Re-orient the (u,v) basis:
        uv_i = tangent_bases  # (N, 2, 3)
        u_i, v_i = uv_i[:, 0, :], uv_i[:, 1, :]  # (N, 3)
        tangent_bases = torch.cat(
            (ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), dim=1
        ).contiguous()  # (N, 6)

        # 4. Store the local 3D frame as an attribute --------------------------
        self.nuv = torch.cat(
            (normals.view(-1, 1, 3), tangent_bases.view(-1, 2, 3)), dim=1
        )


class dMaSIF(nn.Module):
    def __init__(self, params, return_curvatures=True):
        super(dMaSIF, self).__init__()
        if type(params) is not dict:
            params = vars(params)

        self.no_geom = params['no_geom']
        self.no_chem = params['no_chem']
        self.use_mesh = False
        self.return_curvatures = return_curvatures

        # Additional geometric features: mean and Gauss curvatures computed at
        # different scales.
        self.curvature_scales = params['curvature_scales']

        I = params['atom_dims'] + 2 * len(self.curvature_scales)
        O = params['orientation_units']
        H = params.get('hidden_dims', O)
        E = params['emb_dims']

        # Computes chemical features
        self.atomnet = AtomNet_MP(params['atom_dims'])
        self.dropout = nn.Dropout(params['dropout'])

        # Post-processing, without batch norm:
        self.orientation_scores = nn.Sequential(
            nn.Linear(I, O),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(O, 1),
        )

        # Segmentation network:
        self.conv = dMaSIFConv_seg(
            in_channels=I,
            hidden_channels=H,
            out_channels=E,
            n_layers=params['n_layers'],
            radius=params['radius'],
        )

    def features(self, protein):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""

        # Estimate the curvatures using the triangles or the estimated normals:
        curv_feats = curvatures(
            protein["xyz"],
            triangles=protein["triangles"] if self.use_mesh else None,
            normals=None if self.use_mesh else protein["normals"],
            scales=self.curvature_scales,
            batch=protein["batch"],
        )

        # Compute chemical features on-the-fly:
        chem_feats = self.atomnet(
            protein["xyz"], protein["atom_xyz"], protein["atomtypes"],
            protein["batch"], protein["batch_atoms"]
        )

        if self.no_chem:
            chem_feats = 0.0 * chem_feats
        if self.no_geom:
            curv_feats = 0.0 * curv_feats

        # Concatenate our features:
        return torch.cat([curv_feats, chem_feats], dim=1).contiguous(), curv_feats

    def forward(self, protein):

        if 'batch' not in protein:
            protein['batch'] = torch.zeros(len(protein['xyz']), dtype=int, device=protein['xyz'].device)

        # CPU time avg: 3 seconds
        features, curv_feats = self.features(protein)
        features = self.dropout(features)

        # CPU time avg: 0.1 seconds
        self.conv.load_mesh(
            protein["xyz"],
            triangles=protein["triangles"] if self.use_mesh else None,
            normals=None if self.use_mesh else protein["normals"],
            weights=self.orientation_scores(features),
            batch=protein["batch"],
        )

        # CPU time avg: 7 seconds
        out = self.conv(features)

        if not self.return_curvatures:
            return out, protein['normals'].unsqueeze(1)
        return out, curv_feats
