from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latentfrag.encoder.dmasif.helper import ranges_slices

# Maximum number of source points to process at once in pairwise operations.
# Reduces peak GPU memory from O(N²) to O(N × _CHUNK_SIZE).
_CHUNK_SIZE = 2048


def _batch_mask(batch_x, batch_y=None):
    """Build a boolean (N, M) batch-diagonal mask from batch vectors.

    Returns None when no batching is needed.
    """
    if batch_x is None:
        return None
    if batch_y is None:
        batch_y = batch_x
    return batch_x[:, None] == batch_y[None, :]  # (N, M)


def grid_cluster(x, size):
    """Distribute points into cubic bins (reimplemented from pykeops).

    Args:
        x ((M,D) Tensor): point positions.
        size (float): side length of cubic cells.

    Returns:
        (M,) int32 Tensor of cluster labels.
    """
    with torch.no_grad():
        if x.shape[1] == 1:
            weights = torch.IntTensor([1]).to(x.device)
        elif x.shape[1] == 2:
            weights = torch.IntTensor([2 ** 10, 1]).to(x.device)
        elif x.shape[1] == 3:
            weights = torch.IntTensor([2 ** 20, 2 ** 10, 1]).to(x.device)
        else:
            raise NotImplementedError()
        x_ = (x / size).floor().int()
        x_ *= weights
        lab = x_.sum(1)
        lab = lab - lab.min()

        u_lab = torch.unique(lab).sort()[0]
        N_lab = len(u_lab)
        foo = torch.empty(u_lab.max() + 1, dtype=torch.int32, device=x.device)
        foo[u_lab] = torch.arange(N_lab, dtype=torch.int32, device=x.device)
        lab = foo[lab]

    return lab


# On-the-fly generation of the surfaces ========================================


def subsample(x, batch=None, scale=1.0):
    """Subsamples the point cloud using a grid (cubic) clustering scheme.

    The function returns one average sample per cell, as described in Fig. 3.e)
    of the paper.

    Args:
        x (Tensor): (N,3) point cloud.
        batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric.
            Defaults to None.
        scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).

    Returns:
        (M,3): sub-sampled point cloud, with M <= N.
    """

    if batch is None:  # Single protein case:
        if True:  # Use a fast scatter_add_ implementation
            labels = grid_cluster(x, scale).long()
            C = labels.max() + 1

            # We append a "1" to the input vectors, in order to
            # compute both the numerator and denominator of the "average"
            # fraction in one pass through the data.
            x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
            D = x_1.shape[1]
            points = torch.zeros_like(x_1[:C])
            points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
            return (points[:, :-1] / points[:, -1:]).contiguous()

        else:  # Older implementation;
            points = scatter(points * weights[:, None], labels, dim=0)
            weights = scatter(weights, labels, dim=0)
            points = points / weights[:, None]

    else:  # We process proteins using a for loop.
        # This is probably sub-optimal, but I don't really know
        # how to do more elegantly (this type of computation is
        # not super well supported by PyTorch).
        batch_size = torch.max(batch).item() + 1  # Typically, =32
        points, batches = [], []
        for b in range(batch_size):
            p = subsample(x[batch == b], scale=scale)
            points.append(p)
            batches.append(b * torch.ones_like(batch[: len(p)]))

    return torch.cat(points, dim=0), torch.cat(batches, dim=0)


def soft_distances(x, y, batch_x, batch_y, smoothness=0.01, atomtypes=None):
    """Computes a soft distance function to the atom centers of a protein.

    Implements Eq. (1) of the paper in a fast and numerically stable way.

    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
        batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
        smoothness (float, optional): atom radii if atom types are not provided. Defaults to .01.
        atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.

    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    N = x.shape[0]  # atoms
    M = y.shape[0]  # sampling locations

    if atomtypes is not None:
        # Turn the one-hot encoding "atomtypes" into a vector of diameters "smoothness_i":
        atomic_radii = torch.cuda.FloatTensor(
            [170, 110, 152, 155, 180, 190], device=x.device
        )
        atomic_radii = atomic_radii / atomic_radii.min()
        atomtype_radii = atomtypes * atomic_radii[None, :]
        smoothness_vec = torch.sum(
            smoothness * atomtype_radii, dim=1, keepdim=False
        )  # (N,)

        # Chunked accumulation over atoms to avoid O(N*M) memory
        denom = torch.zeros(M, device=y.device, dtype=y.dtype)
        numer = torch.zeros(M, device=y.device, dtype=y.dtype)
        lse_accum = torch.full((M,), float('-inf'), device=y.device, dtype=y.dtype)

        for i0 in range(0, N, _CHUNK_SIZE):
            i1 = min(i0 + _CHUNK_SIZE, N)
            D_c = ((x[i0:i1, None, :] - y[None, :, :]) ** 2).sum(-1)  # (C, M)
            neg_sqrt = -D_c.sqrt()
            exp_c = neg_sqrt.exp()
            s_c = smoothness_vec[i0:i1, None]  # (C, 1)

            if batch_x is not None:
                mask_c = (batch_x[i0:i1, None] == batch_y[None, :])  # (C, M)
                exp_c = exp_c * mask_c.float()

            denom = denom + exp_c.sum(0)
            numer = numer + (s_c * exp_c).sum(0)

            val_c = neg_sqrt / s_c  # (C, M)
            if batch_x is not None:
                val_c = val_c.masked_fill(~mask_c, float('-inf'))
            lse_accum = torch.logaddexp(lse_accum, val_c.logsumexp(dim=0))

        mean_smoothness = numer / denom
        soft_dists = -mean_smoothness * lse_accum

    else:
        lse_accum = torch.full((M,), float('-inf'), device=y.device, dtype=y.dtype)
        for i0 in range(0, N, _CHUNK_SIZE):
            i1 = min(i0 + _CHUNK_SIZE, N)
            D_c = ((x[i0:i1, None, :] - y[None, :, :]) ** 2).sum(-1)  # (C, M)
            val_c = -D_c.sqrt() / smoothness
            if batch_x is not None:
                mask_c = (batch_x[i0:i1, None] == batch_y[None, :])  # (C, M)
                val_c = val_c.masked_fill(~mask_c, float('-inf'))
            lse_accum = torch.logaddexp(lse_accum, val_c.logsumexp(dim=0))
        soft_dists = -smoothness * lse_accum

    return soft_dists


def atoms_to_points_normals(
    atoms,
    batch,
    distance=1.05,
    smoothness=0.5,
    resolution=1.0,
    nits=4,
    atomtypes=None,
    sup_sampling=20,
    variance=0.1,
):
    """Turns a collection of atoms into an oriented point cloud.

    Sampling algorithm for protein surfaces, described in Fig. 3 of the paper.

    Args:
        atoms (Tensor): (N,3) coordinates of the atom centers `a_k`.
        batch (integer Tensor): (N,) batch vector, as in PyTorch_geometric.
        distance (float, optional): value of the level set to sample from
            the smooth distance function. Defaults to 1.05.
        smoothness (float, optional): radii of the atoms, if atom types are
            not provided. Defaults to 0.5.
        resolution (float, optional): side length of the cubic cells in
            the final sub-sampling pass. Defaults to 1.0.
        nits (int, optional): number of iterations . Defaults to 4.
        atomtypes (Tensor, optional): (N,6) one-hot encoding of the atom
            chemical types. Defaults to None.

    Returns:
        (Tensor): (M,3) coordinates for the surface points `x_i`.
        (Tensor): (M,3) unit normals `n_i`.
        (integer Tensor): (M,) batch vector, as in PyTorch_geometric.
    """
    # a) Parameters for the soft distance function and its level set:
    T = distance

    N, D = atoms.shape
    B = sup_sampling  # Sup-sampling ratio

    # Batch vectors:
    batch_atoms = batch
    batch_z = batch[:, None].repeat(1, B).view(N * B)

    # b) Draw N*B points at random in the neighborhood of our atoms
    z = atoms[:, None, :] + 10 * T * torch.randn(N, B, D).type_as(atoms)
    z = z.view(-1, D)  # (N*B, D)

    # We don't want to backprop through a full network here!
    atoms = atoms.detach().contiguous()
    z = z.detach().contiguous()

    # N.B.: Test mode disables the autograd engine: we must switch it on explicitely.
    with torch.enable_grad():
        if z.is_leaf:
            z.requires_grad = True

        # c) Iterative loop: gradient descent along the potential
        # ".5 * (dist - T)^2" with respect to the positions z of our samples
        for it in range(nits):
            dists = soft_distances(
                atoms,
                z,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            Loss = ((dists - T) ** 2).sum()
            g = torch.autograd.grad(Loss, z)[0]
            z.data -= 0.5 * g

        # d) Only keep the points which are reasonably close to the level set:
        dists = soft_distances(
            atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes
        )
        margin = (dists - T).abs()
        mask = margin < variance * T

        # d') And remove the points that are trapped *inside* the protein:
        zz = z.detach()
        zz.requires_grad = True
        for it in range(nits):
            dists = soft_distances(
                atoms,
                zz,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            Loss = (1.0 * dists).sum()
            g = torch.autograd.grad(Loss, zz)[0]
            normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
            zz = zz + 1.0 * T * normals

        dists = soft_distances(
            atoms, zz, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes
        )
        mask = mask & (dists > 1.5 * T)

        z = z[mask].contiguous().detach()
        batch_z = batch_z[mask].contiguous().detach()

        # e) Subsample the point cloud:
        points, batch_points = subsample(z, batch_z, scale=resolution)

        # f) Compute the normals on this smaller point cloud:
        p = points.detach()
        p.requires_grad = True
        dists = soft_distances(
            atoms,
            p,
            batch_atoms,
            batch_points,
            smoothness=smoothness,
            atomtypes=atomtypes,
        )
        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss, p)[0]
        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
    points = points - 0.5 * normals
    return points.detach(), normals.detach(), batch_points.detach()


# Surface mesh -> Normals ======================================================


def mesh_normals_areas(vertices, triangles=None, scale=[1.0], batch=None, normals=None):
    """Returns a smooth field of normals, possibly at different scales.

    points, triangles or normals, scale(s)  ->      normals
    (N, 3),    (3, T) or (N,3),      (S,)   ->  (N, 3) or (N, S, 3)

    Simply put - if `triangles` are provided:
      1. Normals are first computed for every triangle using simple 3D geometry
         and are weighted according to surface area.
      2. The normal at any given vertex is then computed as the weighted average
         of the normals of all triangles in a neighborhood specified
         by Gaussian windows whose radii are given in the list of "scales".

    If `normals` are provided instead, we simply smooth the discrete vector
    field using Gaussian windows whose radii are given in the list of "scales".

    If more than one scale is provided, normal fields are computed in parallel
    and returned in a single 3D tensor.

    Args:
        vertices (Tensor): (N,3) coordinates of mesh vertices or 3D points.
        triangles (integer Tensor, optional): (3,T) mesh connectivity. Defaults to None.
        scale (list of floats, optional): (S,) radii of the Gaussian smoothing windows. Defaults to [1.].
        batch (integer Tensor, optional): batch vector, as in PyTorch_geometric. Defaults to None.
        normals (Tensor, optional): (N,3) raw normals vectors on the vertices. Defaults to None.

    Returns:
        (Tensor): (N,3) or (N,S,3) point normals.
        (Tensor): (N,) point areas, if triangles were provided.
    """

    # Single- or Multi-scale mode:
    if hasattr(scale, "__len__"):
        scales, single_scale = scale, False
    else:
        scales, single_scale = [scale], True
    scales = torch.Tensor(scales).type_as(vertices)  # (S,)

    # Compute the "raw" field of normals:
    if triangles is not None:
        # Vertices of all triangles in the mesh:
        A = vertices[triangles[0, :]]  # (N, 3)
        B = vertices[triangles[1, :]]  # (N, 3)
        C = vertices[triangles[2, :]]  # (N, 3)

        # Triangle centers and normals (length = surface area):
        centers = (A + B + C) / 3  # (N, 3)
        V = (B - A).cross(C - A)  # (N, 3)

        # Vertice areas:
        S = (V ** 2).sum(-1).sqrt() / 6  # (N,) 1/3 of a triangle area
        areas = torch.zeros(len(vertices)).type_as(vertices)  # (N,)
        areas.scatter_add_(0, triangles[0, :], S)  # Aggregate from "A's"
        areas.scatter_add_(0, triangles[1, :], S)  # Aggregate from "B's"
        areas.scatter_add_(0, triangles[2, :], S)  # Aggregate from "C's"

    else:  # Use "normals" instead
        areas = None
        V = normals
        centers = vertices

    # Normal of a vertex = average of all normals in a ball of size "scale".
    # Chunked over source points to avoid O(N*M) memory.
    Nv = vertices.shape[0]
    M = centers.shape[0]

    if batch is not None:
        batch_vertices = batch
        batch_centers = batch[triangles[0, :]] if triangles is not None else batch
    else:
        batch_vertices = None
        batch_centers = None

    if single_scale:
        U = torch.zeros(Nv, 3, device=vertices.device, dtype=vertices.dtype)
        scale_sq2 = 2 * (scales[0] ** 2)
        for j0 in range(0, M, _CHUNK_SIZE):
            j1 = min(j0 + _CHUNK_SIZE, M)
            D_c = ((vertices[:, None, :] - centers[None, j0:j1, :]) ** 2).sum(-1)  # (Nv, chunk)
            K_c = (-D_c / scale_sq2).exp()
            if batch_vertices is not None:
                mask_c = (batch_vertices[:, None] == batch_centers[None, j0:j1])
                K_c = K_c * mask_c.float()
            U = U + (K_c[:, :, None] * V[None, j0:j1, :]).sum(dim=1)
    else:
        S = len(scales)
        U = torch.zeros(Nv, S, 3, device=vertices.device, dtype=vertices.dtype)
        for j0 in range(0, M, _CHUNK_SIZE):
            j1 = min(j0 + _CHUNK_SIZE, M)
            D_c = ((vertices[:, None, :] - centers[None, j0:j1, :]) ** 2).sum(-1)  # (Nv, chunk)
            if batch_vertices is not None:
                mask_c = (batch_vertices[:, None] == batch_centers[None, j0:j1])
            V_c = V[j0:j1, :]
            for s_idx in range(S):
                K_s = (-D_c / (2 * scales[s_idx] ** 2)).exp()
                if batch_vertices is not None:
                    K_s = K_s * mask_c.float()
                U[:, s_idx, :] = U[:, s_idx, :] + (K_s[:, :, None] * V_c[None, :, :]).sum(dim=1)

    normals = F.normalize(U, p=2, dim=-1)  # (Nv, 3) or (Nv, S, 3)

    return normals, areas


# Compute tangent planes and curvatures ========================================


def tangent_vectors(normals):
    """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].

          normals        ->             uv
    (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)

    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".

    Args:
        normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

    Returns:
        (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
            the tangent coordinate systems `[n_i,u_i,v_i].
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv


def curvatures(
    vertices, triangles=None, scales=[1.0], batch=None, normals=None, reg=0.01
):
    """Returns a collection of mean (H) and Gauss (K) curvatures at different scales.

    points, faces, scales  ->  (H_1, K_1, ..., H_S, K_S)
    (N, 3), (3, N), (S,)   ->         (N, S*2)

    We rely on a very simple linear regression method, for all vertices:

      1. Estimate normals and surface areas.
      2. Compute a local tangent frame.
      3. In a pseudo-geodesic Gaussian neighborhood at scale s,
         compute the two (2, 2) covariance matrices PPt and PQt
         between the displacement vectors "P = x_i - x_j" and
         the normals "Q = n_i - n_j", projected on the local tangent plane.
      4. Up to the sign, the shape operator S at scale s is then approximated
         as  "S = (reg**2 * I_2 + PPt)^-1 @ PQt".
      5. The mean and Gauss curvatures are the trace and determinant of
         this (2, 2) matrix.

    As of today, this implementation does not weigh points by surface areas:
    this could make a sizeable difference if protein surfaces were not
    sub-sampled to ensure uniform sampling density.

    For convergence analysis, see for instance
    "Efficient curvature estimation for oriented point clouds",
    Cao, Li, Sun, Assadi, Zhang, 2019.

    Args:
        vertices (Tensor): (N,3) coordinates of the points or mesh vertices.
        triangles (integer Tensor, optional): (3,T) mesh connectivity. Defaults to None.
        scales (list of floats, optional): list of (S,) smoothing scales. Defaults to [1.].
        batch (integer Tensor, optional): batch vector, as in PyTorch_geometric. Defaults to None.
        normals (Tensor, optional): (N,3) field of "raw" unit normals. Defaults to None.
        reg (float, optional): small amount of Tikhonov/ridge regularization
            in the estimation of the shape operator. Defaults to .01.

    Returns:
        (Tensor): (N, S*2) tensor of mean and Gauss curvatures computed for
            every point at the required scales.
    """
    # Number of points, number of scales:
    N, S = vertices.shape[0], len(scales)
    # Compute the normals at different scales + vertice areas:
    normals_s, _ = mesh_normals_areas(
        vertices, triangles=triangles, normals=normals, scale=scales, batch=batch
    )  # (N, S, 3), (N,)

    # Local tangent bases:
    uv_s = tangent_vectors(normals_s)  # (N, S, 2, 3)

    features = []

    for s, scale in enumerate(scales):
        # Extract the relevant descriptors at the current scale:
        normals = normals_s[:, s, :].contiguous()  # (N, 3)
        uv = uv_s[:, s, :, :].contiguous()  # (N, 2, 3)

        # Chunked aggregation over j to avoid O(N²) memory
        scale_sq2 = 2 * (scale ** 2)
        PPt_PQt = torch.zeros(N, 8, device=vertices.device, dtype=vertices.dtype)

        for j0 in range(0, N, _CHUNK_SIZE):
            j1 = min(j0 + _CHUNK_SIZE, N)
            vj = vertices[j0:j1, :]  # (chunk, 3)
            nj = normals[j0:j1, :]   # (chunk, 3)

            diff_x = vj[None, :, :] - vertices[:, None, :]  # (N, chunk, 3)
            diff_n = nj[None, :, :] - normals[:, None, :]    # (N, chunk, 3)

            sq_dist = (diff_x ** 2).sum(-1)  # (N, chunk)
            dot_nn = (normals[:, None, :] * nj[None, :, :]).sum(-1)  # (N, chunk)
            d2_c = sq_dist * ((2 - dot_nn) ** 2)
            window_c = (-d2_c / scale_sq2).exp()  # (N, chunk)

            P_c = torch.einsum('iac,ijc->ija', uv, diff_x)  # (N, chunk, 2)
            Q_c = torch.einsum('iac,ijc->ija', uv, diff_n)  # (N, chunk, 2)
            PQ_c = torch.cat([P_c, Q_c], dim=-1)  # (N, chunk, 4)

            PPQ_c = P_c[:, :, :, None] * PQ_c[:, :, None, :]  # (N, chunk, 2, 4)
            C = j1 - j0
            PPQ_c = PPQ_c.reshape(N, C, 8)
            PPQ_c = window_c[:, :, None] * PPQ_c

            if batch is not None:
                mask_c = (batch[:, None] == batch[None, j0:j1])  # (N, chunk)
                PPQ_c = PPQ_c * mask_c[:, :, None].float()

            PPt_PQt = PPt_PQt + PPQ_c.sum(1)  # (N, 8)

        # Reshape to get the two covariance matrices:
        PPt_PQt = PPt_PQt.view(N, 2, 2, 2)
        PPt, PQt = PPt_PQt[:, :, 0, :], PPt_PQt[:, :, 1, :]  # (N, 2, 2), (N, 2, 2)

        # Add a small ridge regression:
        PPt[:, 0, 0] += reg
        PPt[:, 1, 1] += reg

        # (minus) Shape operator, i.e. the differential of the Gauss map:
        # = (PPt^-1 @ PQt) : simple estimation through linear regression
        # S = torch.solve(PQt, PPt).solution
        S = torch.linalg.solve(PPt, PQt)
        a, b, c, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]  # (N,)

        # Normalization
        mean_curvature = a + d
        gauss_curvature = a * d - b * c
        features += [mean_curvature.clamp(-1, 1), gauss_curvature.clamp(-1, 1)]

    features = torch.stack(features, dim=-1)
    return features


# Fast tangent convolution layer ===============================================


class dMaSIFConv(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, radius=1.0, hidden_units=None, cheap=False
    ):
        """Creates the KeOps convolution layer.

        I = in_channels  is the dimension of the input features
        O = out_channels is the dimension of the output features
        H = hidden_units is the dimension of the intermediate representation
        radius is the size of the pseudo-geodesic Gaussian window w_ij = W(d_ij)


        This affordable layer implements an elementary "convolution" operator
        on a cloud of N points (x_i) in dimension 3 that we decompose in three steps:

          1. Apply the MLP "net_in" on the input features "f_i". (N, I) -> (N, H)

          2. Compute H interaction terms in parallel with:
                  f_i = sum_j [ w_ij * conv(P_ij) * f_j ]
            In the equation above:
              - w_ij is a pseudo-geodesic window with a set radius.
              - P_ij is a vector of dimension 3, equal to "x_j-x_i"
                in the local oriented basis at x_i.
              - "conv" is an MLP from R^3 to R^H:
                 - with 1 linear layer if "cheap" is True;
                 - with 2 linear layers and C=8 intermediate "cuts" otherwise.
              - "*" is coordinate-wise product.
              - f_j is the vector of transformed features.

          3. Apply the MLP "net_out" on the output features. (N, H) -> (N, O)


        A more general layer would have implemented conv(P_ij) as a full
        (H, H) matrix instead of a mere (H,) vector... At a much higher
        computational cost. The reasoning behind the code below is that
        a given time budget is better spent on using a larger architecture
        and more channels than on a very complex convolution operator.
        Interactions between channels happen at steps 1. and 3.,
        whereas the (costly) point-to-point interaction step 2.
        lets the network aggregate information in spatial neighborhoods.

        Args:
            in_channels (int, optional): numper of input features per point. Defaults to 1.
            out_channels (int, optional): number of output features per point. Defaults to 1.
            radius (float, optional): deviation of the Gaussian window on the
                quasi-geodesic distance `d_ij`. Defaults to 1..
            hidden_units (int, optional): number of hidden features per point.
                Defaults to out_channels.
            cheap (bool, optional): shall we use a 1-layer deep Filter,
                instead of a 2-layer deep MLP? Defaults to False.
        """

        super(dMaSIFConv, self).__init__()

        self.Input = in_channels
        self.Output = out_channels
        self.Radius = radius
        self.Hidden = self.Output if hidden_units is None else hidden_units
        self.Cuts = 8  # Number of hidden units for the 3D MLP Filter.
        self.cheap = cheap

        # For performance reasons, we cut our "hidden" vectors
        # in n_heads "independent heads" of dimension 8.
        self.heads_dim = 8  # 4 is probably too small; 16 is certainly too big

        # We accept "Hidden" dimensions of size 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, ...
        if self.Hidden < self.heads_dim:
            self.heads_dim = self.Hidden

        if self.Hidden % self.heads_dim != 0:
            raise ValueError(f"The dimension of the hidden units ({self.Hidden})"\
                    + f"should be a multiple of the heads dimension ({self.heads_dim}).")
        else:
            self.n_heads = self.Hidden // self.heads_dim

        # Transformation of the input features:
        self.net_in = nn.Sequential(
            nn.Linear(self.Input, self.Hidden),  # (H, I) + (H,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Hidden, self.Hidden),  # (H, H) + (H,)
            # nn.LayerNorm(self.Hidden),#nn.BatchNorm1d(self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
        )  # (H,)
        self.norm_in = nn.GroupNorm(4, self.Hidden)
        # self.norm_in = nn.LayerNorm(self.Hidden)
        # self.norm_in = nn.Identity()

        # 3D convolution filters, encoded as an MLP:
        if cheap:
            self.conv = nn.Sequential(
                nn.Linear(3, self.Hidden), nn.ReLU()  # (H, 3) + (H,)
            )  # KeOps does not support well LeakyReLu
        else:
            self.conv = nn.Sequential(
                nn.Linear(3, self.Cuts),  # (C, 3) + (C,)
                nn.ReLU(),  # KeOps does not support well LeakyReLu
                nn.Linear(self.Cuts, self.Hidden),
            )  # (H, C) + (H,)

        # Transformation of the output features:
        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Output, self.Output),  # (O, O) + (O,)
            # nn.LayerNorm(self.Output),#nn.BatchNorm1d(self.Output),
            nn.LeakyReLU(negative_slope=0.2),
        )  # (O,)

        self.norm_out = nn.GroupNorm(4, self.Output)
        # self.norm_out = nn.LayerNorm(self.Output)
        # self.norm_out = nn.Identity()

        # Custom initialization for the MLP convolution filters:
        # we get interesting piecewise affine cuts on a normalized neighborhood.
        with torch.no_grad():
            nn.init.normal_(self.conv[0].weight)
            nn.init.uniform_(self.conv[0].bias)
            self.conv[0].bias *= 0.8 * (self.conv[0].weight ** 2).sum(-1).sqrt()

            if not cheap:
                nn.init.uniform_(
                    self.conv[2].weight,
                    a=-1 / np.sqrt(self.Cuts),
                    b=1 / np.sqrt(self.Cuts),
                )
                nn.init.normal_(self.conv[2].bias)
                self.conv[2].bias *= 0.5 * (self.conv[2].weight ** 2).sum(-1).sqrt()

    def forward(self, points, nuv, features, ranges=None):
        """Performs a quasi-geodesic interaction step.

        points, local basis, in features  ->  out features
        (N, 3),   (N, 3, 3),    (N, I)    ->    (N, O)

        This layer computes the interaction step of Eq. (7) in the paper,
        in-between the application of two MLP networks independently on all
        feature vectors.

        Args:
            points (Tensor): (N,3) point coordinates `x_i`.
            nuv (Tensor): (N,3,3) local coordinate systems `[n_i,u_i,v_i]`.
            features (Tensor): (N,I) input feature vectors `f_i`.
            ranges (6-uple of integer Tensors, optional): batch ranges (kept
                for API compatibility). When provided, the first element is
                used to extract batch boundaries for masking. Defaults to None.

        Returns:
            (Tensor): (N,O) output feature vectors `f'_i`.
        """

        # 1. Transform the input features: -------------------------------------
        features = self.net_in(features)  # (N, I) -> (N, H)
        features = features.transpose(1, 0)[None, :, :]  # (1,H,N)
        features = self.norm_in(features)
        features = features[0].transpose(1, 0).contiguous()  # (1, H, N) -> (N, H)

        # 2. Compute the local "shape contexts": -------------------------------
        N = points.shape[0]
        H = features.shape[1]

        # 2.a Normalize the kernel radius:
        points = points / (sqrt(2.0) * self.Radius)  # (N, 3)

        # WARNING - Here, we assume that the normals are fixed:
        normals = (
            nuv[:, 0, :].contiguous().detach()
        )  # (N, 3) - remove the .detach() if needed
        nuv_matrix = nuv.view(N, 3, 3)  # (N, 3, 3)

        # Derive per-point batch assignment from ranges for chunked masking:
        if ranges is not None:
            range_tensor = ranges[0]  # (B, 2) int tensor
            point_batch = torch.zeros(N, dtype=torch.long, device=points.device)
            for b_idx in range(range_tensor.shape[0]):
                s = range_tensor[b_idx, 0].item()
                e = range_tensor[b_idx, 1].item()
                point_batch[s:e] = b_idx
        else:
            point_batch = None

        # Extract MLP parameters once:
        if self.cheap:
            A = self.conv[0].weight    # (H, 3)
            B_bias = self.conv[0].bias  # (H,)
        else:
            A_1 = self.conv[0].weight  # (C, 3)
            B_1 = self.conv[0].bias    # (C,)
            A_2 = self.conv[2].weight  # (H, C)
            B_2 = self.conv[2].bias    # (H,)

        # Chunked aggregation over j (source points) to avoid O(N²) memory:
        result = torch.zeros(N, H, device=points.device, dtype=features.dtype)

        for j0 in range(0, N, _CHUNK_SIZE):
            j1 = min(j0 + _CHUNK_SIZE, N)
            pj = points[j0:j1, :]   # (chunk, 3)
            nj = normals[j0:j1, :]  # (chunk, 3)

            diff = pj[None, :, :] - points[:, None, :]  # (N, chunk, 3)
            sq_dist = (diff ** 2).sum(-1)  # (N, chunk)
            dot_nn = (normals[:, None, :] * nj[None, :, :]).sum(-1)  # (N, chunk)
            d2 = sq_dist * ((2 - dot_nn) ** 2)
            window = (-d2).exp()  # (N, chunk)

            X_ij = torch.einsum('iac,ijc->ija', nuv_matrix, diff)  # (N, chunk, 3)

            if self.cheap:
                X_conv = torch.einsum('ijk,hk->ijh', X_ij, A) + B_bias[None, None, :]
                X_conv = X_conv.relu()  # (N, chunk, H)
            else:
                X_conv = torch.einsum('ijk,ck->ijc', X_ij, A_1) + B_1[None, None, :]
                X_conv = X_conv.relu()
                X_conv = torch.einsum('ijc,hc->ijh', X_conv, A_2) + B_2[None, None, :]
                X_conv = X_conv.relu()  # (N, chunk, H)

            f_j = features[j0:j1, :][None, :, :]  # (1, chunk, H)
            F_ij = window[:, :, None] * X_conv * f_j  # (N, chunk, H)

            if point_batch is not None:
                mask_c = (point_batch[:, None] == point_batch[None, j0:j1])  # (N, chunk)
                F_ij = F_ij * mask_c[:, :, None].float()

            result = result + F_ij.sum(dim=1)

        features = result  # (N, H)

        # 3. Transform the output features: ------------------------------------
        features = self.net_out(features)  # (N, H) -> (N, O)
        features = features.transpose(1, 0)[None, :, :]  # (1,O,N)
        features = self.norm_out(features)
        features = features[0].transpose(1, 0).contiguous()

        return features
