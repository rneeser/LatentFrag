from abc import ABC
from abc import abstractmethod

import torch
from torch_scatter import scatter_mean, scatter_add

import latentfrag.fm.utils.so3_utils as so3


class ICFM(ABC):
    """
    Abstract base class for all Independent-coupling CFM classes.
    Defines a common interface.
    Notation:
    - zt is the intermediate representation at time step t \in [0, 1]
    - zs is the noised representation at time step s < t

    # TODO: add interpolation schedule (not necessrily linear)
    """
    def __init__(self, sigma):
        self.sigma = sigma

    @abstractmethod
    def sample_zt(self, z0, z1, t, *args, **kwargs):
        """ TODO. """
        pass

    @abstractmethod
    def sample_zt_given_zs(self, *args, **kwargs):
        """ Perform update, typically using an explicit Euler step. """
        pass

    @abstractmethod
    def sample_z0(self, *args, **kwargs):
        """ Prior. """
        pass

    @abstractmethod
    def compute_loss(self, pred, z0, z1, *args, **kwargs):
        """ Compute loss per sample. """
        pass


class CoordICFM(ICFM):
    def __init__(self, sigma):
        self.dim = 3
        # scaling factor: model has to predict vector of length 1 with it
        self.scale = 2.7
        super().__init__(sigma)

    def sample_zt(self, z0, z1, t, batch_mask):
        zt = t[batch_mask] * z1 + (1 - t)[batch_mask] * z0
        # zt = self.sigma * z0 + t[batch_mask] * z1 + (1 - t)[batch_mask] * z0  # TODO: do we have to compute Psi?
        return zt

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform an explicit Euler step. """
        step_size = t - s
        zt = zs + step_size[batch_mask] * self.scale * pred
        return zt

    def sample_z0(self, com, batch_mask):
        """ Prior. """
        z0 = torch.randn((len(batch_mask), self.dim), device=batch_mask.device)

        # Move center of mass
        z0 = z0 + com[batch_mask]

        return z0

    def compute_loss(self, pred, z0, z1, t, batch_mask):
        """ Compute loss per sample. """
        loss = torch.sum((pred - (z1 - z0) / self.scale) ** 2, dim=-1) / self.dim
        loss = scatter_mean(loss, batch_mask, dim=0)
        return loss

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """
        # z1 = z0 + pred
        z1 = zt + (1 - t)[batch_mask] * pred
        return z1


class SO3ICFM(ICFM):
    """
    All rotations are assumed to be in axis-angle format.
    Mostly following descriptions from the FoldFlow paper:
    https://openreview.net/forum?id=kJFIH23hXb

    See also:
    https://geomstats.github.io/_modules/geomstats/geometry/special_orthogonal.html#SpecialOrthogonal
    https://geomstats.github.io/_modules/geomstats/geometry/lie_group.html#LieGroup
    """
    def __init__(self, sigma):
        super().__init__(sigma)

    def exponential_map(self, base, tangent):
        """
        Args:
            base: base point (rotation vector) on the manifold
            tangent: point in tangent space at identity
        Returns:
            rotation vector on the manifold
        """
        # return so3.exp_not_from_identity(tangent, base_point=base)
        return so3.compose_rotations(base, so3.exp(tangent))

    def logarithm_map(self, base, r):
        """
        Args:
            base: base point (rotation vector) on the manifold
            r: rotation vector on the manifold
        Return:
            point in tangent space at identity
        """
        # return so3.log_not_from_identity(r, base_point=base)
        return so3.log(so3.compose_rotations(-base, r))

    def sample_zt(self, z0, z1, t, batch_mask):
        """
        Expressed in terms of exponential and logarithm maps.
        Corresponds to SLERP interpolation: R(t) = R1 exp( t * log(R1^T R2) )
        (see https://lucaballan.altervista.org/pdfs/IK.pdf, slide 16)
        """

        # apply logarithm map
        zt_tangent = t[batch_mask] * self.logarithm_map(z0, z1)

        # apply exponential map
        return self.exponential_map(z0, zt_tangent)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        # estimate z1_tangent based on zt and pred only
        z1_tangent = (1 - t)[batch_mask] * pred

        # exponential map
        return self.exponential_map(zt, z1_tangent)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        # # parallel transport vector field to lie algebra so3 (at identity)
        # # (FoldFlow paper, Algorithm 3, line 8)
        # # TODO: is this correct? is it necessary?
        # pred = so3.compose(so3.inverse(zs), pred)

        step_size = t - s
        zt_tangent = step_size[batch_mask] * pred

        # exponential map
        return self.exponential_map(zs, zt_tangent)

    def sample_z0(self, batch_mask):
        """ Prior. """
        return so3.random_uniform(n_samples=len(batch_mask), device=batch_mask.device)

    @staticmethod
    def d_R_squared_SO3(rot_vec_1, rot_vec_2):
        """
        Squared Riemannian metric on SO(3).
        Defined as d(R1, R2) = sqrt(0.5) ||log(R1^T R2)||_F
        where R1, R2 are rotation matrices.

        The following is equivalent if the difference between the rotations is
        expressed as a rotation vector \omega_diff:
        d(r1, r2) = ||\omega_diff||_2
        -----
        With the definition of the Frobenius matrix norm ||A||_F^2 = trace(A^H A):
        d^2(R1, R2) = 1/2 ||log(R1^T R2)||_F^2
                    = 1/2 || hat(R_d) ||_F^2
                    = 1/2 tr( hat(R_d)^T hat(R_d) )
                    = 1/2 * 2 * ||\omega||_2^2
        """

        # rot_mat_1 = so3.matrix_from_rotation_vector(rot_vec_1)
        # rot_mat_2 = so3.matrix_from_rotation_vector(rot_vec_2)
        # rot_mat_diff = rot_mat_1.transpose(-2, -1) @ rot_mat_2
        # return torch.norm(so3.log(rot_mat_diff, as_skew=True), p='fro', dim=(-2, -1))

        diff_rot = so3.compose_rotations(-rot_vec_1, rot_vec_2)
        return diff_rot.square().sum(dim=-1)

    def compute_loss(self, pred, z1, zt, t, batch_mask, reduce='mean', eps=5e-2):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}

        zt_dot = self.logarithm_map(zt, z1) / torch.clamp(1 - t, min=eps)[batch_mask]

        loss = torch.sum((pred - zt_dot)**2, dim=-1)
        # loss = self.d_R_squared_SO3(zt_dot, pred)

        if reduce == 'mean':
            loss = scatter_mean(loss, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss


class SphericalICFM(ICFM):
    """
    Spherical flow matching
    """
    def __init__(self, sigma, dim, scale):
        self.dim = dim
        # scaling factor for tangent space operations
        self.scale = scale  # TODO should be based on mean/median angual distances between embeddings
        super().__init__(sigma)

    def project_to_sphere(self, x):
        """Project points onto unit sphere."""
        return x / torch.norm(x, dim=-1, keepdim=True)

    def log_map(self, x, base):
        """
        Logarithmic map from sphere to tangent space at base point.
        v = θ/sin(θ) * (x - cos(θ)base)
        """
        # Compute inner product = cosine of angle
        # clamp for numerical stability
        inner = torch.sum(base * x, dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        # angle between vectors (geodesic distance)
        theta = torch.arccos(inner)
        # scaling factor
        factor = torch.where(theta > 1e-7,
                    theta / torch.sin(theta),
                    1.0 + theta**2/6)
        # tangent vector
        return factor * (x - inner * base)

    def exp_map(self, v, base):
        """
        Exponential map from tangent space at base point to sphere.
        exp_p(v) = cos(||v||)p + sin(||v||)(v/||v||)
        """
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        # Handle zero-norm case
        norm_v = torch.where(norm_v < 1e-7,
                           torch.ones_like(norm_v),
                           norm_v)

        cos_theta = torch.cos(norm_v)
        sin_theta = torch.sin(norm_v)

        return cos_theta * base + sin_theta * (v / norm_v)

    def sample_zt(self, z0, z1, t, batch_mask):
        """
        Spherical interpolation (SLERP) between z0 and z1.
        SLERP(z0, z1, t) = sin((1-t)θ)/sin(θ) * z0 + sin(tθ)/sin(θ) * z1
        """
        # Ensure inputs are on sphere
        z0_norm = self.project_to_sphere(z0)
        z1_norm = self.project_to_sphere(z1)

        # Compute the angle between vectors
        cos_omega = torch.sum(z0_norm * z1_norm, dim=-1, keepdim=True)
        cos_omega = cos_omega.clamp(-1 + 1e-7, 1 - 1e-7)
        omega = torch.arccos(cos_omega)

        # Perform spherical interpolation
        sin_omega = torch.sin(omega)

        # Handle small angle case
        mask = sin_omega < 1e-6

        zt = torch.where(mask,
                        z0_norm + t[batch_mask] * (z1_norm - z0_norm),
                        (torch.sin((1 - t[batch_mask]) * omega) / sin_omega) * z0_norm +
                        (torch.sin(t[batch_mask] * omega) / sin_omega) * z1_norm)

        # Ensure output is normalized
        return self.project_to_sphere(zt)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """Perform an explicit Euler step in the tangent space."""
        step_size = t - s
        # Move in tangent space
        zs_norm = self.project_to_sphere(zs)
        tangent_step = step_size[batch_mask] * self.scale * pred
        # Use exponential map to move back to sphere
        zt = self.exp_map(tangent_step, zs_norm)
        return zt

    def sample_z0(self, batch_mask):
        """Sample uniformly from the sphere."""
        # Sample from standard normal
        z0 = torch.randn((len(batch_mask), self.dim), device=batch_mask.device)
        # Project to sphere - this gives uniform distribution because of the
        # rotational invariance of the normal distribution
        z0 = self.project_to_sphere(z0)
        return z0

    def compute_loss(self, pred, z0, z1, t, batch_mask):
        """Compute loss in tangent space."""
        z0_norm = self.project_to_sphere(z0)
        z1_norm = self.project_to_sphere(z1)

        # Convert target to tangent space
        target = self.log_map(z1_norm, z0_norm) / self.scale

        # Compute MSE in tangent space
        loss = torch.sum((pred - target) ** 2, dim=-1) / self.dim
        loss = scatter_mean(loss, batch_mask, dim=0)
        return loss

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """Predict final state using exponential map."""
        zt_norm = self.project_to_sphere(zt)
        tangent_vec = (1 - t)[batch_mask] * pred
        z1 = self.exp_map(self.scale * tangent_vec, zt_norm)
        return z1


class FragEmbedICFM(ICFM):
    def __init__(self, sigma, dim, scale=0.2):
        self.dim = dim
        self.scale = scale  # approx. avg. norm of embeddings
        super().__init__(sigma)

    def sample_zt(self, z0, z1, t, batch_mask):
        zt = t[batch_mask] * z1 + (1 - t)[batch_mask] * z0
        return zt

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform an explicit Euler step. """
        step_size = t - s
        zt = zs + step_size[batch_mask] * self.scale * pred
        return zt

    def sample_z0(self, batch_mask):
        """ Prior. """
        z0 = torch.randn((len(batch_mask), self.dim), device=batch_mask.device)
        return z0

    def compute_loss(self, pred, z0, z1, t, batch_mask):
        """ Compute loss per sample. """
        loss = torch.sum((pred - (z1 - z0) / self.scale) ** 2, dim=-1) / self.dim
        loss = scatter_mean(loss, batch_mask, dim=0)
        return loss

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """
        z1 = zt + (1 - t)[batch_mask] * pred
        return z1


class MarginalFragEmbedICFM(FragEmbedICFM):
    def __init__(self, sigma, dim, scale, mean_p, std_p):
        # prior: means and variances for every dimension
        self.mean_p = torch.from_numpy(mean_p)
        self.std_p = torch.from_numpy(std_p)
        super().__init__(sigma, dim, scale)

    def sample_z0(self, batch_mask):
        """ Prior. Sampling from known multivariate Gaussian. """
        z0 = torch.randn((len(batch_mask), self.dim), device=batch_mask.device)
        z0 = z0 * self.std_p.to(batch_mask.device) + self.mean_p.to(batch_mask.device)
        return z0