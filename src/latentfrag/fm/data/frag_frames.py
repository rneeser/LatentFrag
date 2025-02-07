import numpy as np
from sklearn.decomposition import PCA
import torch

import latentfrag.fm.utils.so3_utils as so3
from latentfrag.fm.data.fragment import clean_arom_fragment


def local_frame_from_vecs(exit_vectors):
    """
    Generate a local frame for a fragment based on its internal vectors (exit vectors or COM-atoms) using PCA.
    Returns frame as rotation matrix i.e. stack of x, y, z axes (frame[0] != x-axis; frame.T[0] == x-axis). Recomputing after rotating with one or two exit vectors may give different results as it is underdetermined.

    :param exit_vectors: numpy array of shape (n, 3) where n is the number of exit vectors
    :return: numpy array of shape (3, 3) representing the local frame
    """
    # Ensure we have at least one exit vector and convert to float
    if len(exit_vectors) == 0:
        raise ValueError("At least one exit vector is required")

    exit_vectors = np.array(exit_vectors, dtype=float)

    # If we only have one exit vector, we need to generate a frame differently
    if len(exit_vectors) == 1:
        z_axis = exit_vectors[0] / np.linalg.norm(exit_vectors[0])
        x_axis = np.array([1, 0, 0]) if np.allclose(z_axis, [0, 1, 0]) else np.array([0, 1, 0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        frame = np.column_stack([x_axis, y_axis, z_axis])
        return frame

    # If we have two exit vectors
    elif len(exit_vectors) == 2:
        z_axis = exit_vectors[1] - exit_vectors[0]
        z_axis /= np.linalg.norm(z_axis)
        ref = np.array([1, 0, 0]) if np.allclose(z_axis, [1, 0, 0]) else np.array([0, 1, 0])
        y_axis = np.cross(z_axis, ref)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        frame = np.column_stack([x_axis, y_axis, z_axis])
        return frame

    # If we have three or more exit vectors
    else:
        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(exit_vectors)
        components = pca.components_

        # Ensure right-handed coordinate system
        if np.dot(np.cross(components[0], components[1]), components[2]) < 0:
            components[2] = -components[2]

        frame = np.column_stack(components)

        return frame


def generate_local_frame_ev(m):
    """
    Generate a local frame for a molecule based on its exit vectors.

    :param m: RDKit molecule
    :return: tuple of (origin, x_axis, y_axis, z_axis)
    """
    exit_vectors = get_exit_vectors(m)
    frame = local_frame_from_vecs(exit_vectors)
    return frame


def generate_local_frame_aa(m, use_ev=True):
    """
    Generate a local frame for a molecule based on all its vectors from center to each atom.

    :param m: RDKit molecule
    :return: tuple of (origin, x_axis, y_axis, z_axis)
    """
    if not use_ev:
        m = clean_arom_fragment(m)
    exit_vectors = get_all_vecs(m)
    frame = local_frame_from_vecs(exit_vectors)
    return frame


def get_all_vecs(m):
    """
    Get all vectors from the center of mass of the molecule to each atom.
    """
    coords = m.GetConformer().GetPositions()
    com = coords.mean(axis=0)
    vecs = coords - com

    return vecs


def get_exit_vectors(m):
    """
    Get the normalized exit vectors for a molecule. An exit vecgtor is defined as the vector from an atom to a dummy atom, where it can connect to another fragment.

    :param m: RDKit molecule
    :return: numpy array of shape (n, 3) where n is the number of exit vectors
    """
    num_evs = num_ev(m)
    exit_vectors = np.zeros((num_evs, 3))

    coords = m.GetConformer().GetPositions()
    i = 0
    for b in m.GetBonds():
        if b.GetBeginAtom().GetAtomicNum() == 0:
            exit_vectors[i] = coords[b.GetBeginAtomIdx()] - coords[b.GetEndAtomIdx()]
            i += 1
        elif b.GetEndAtom().GetAtomicNum() == 0:
            exit_vectors[i] = coords[b.GetEndAtomIdx()] - coords[b.GetBeginAtomIdx()]
            i += 1

    # normalize the exit vectors
    exit_vectors /= np.linalg.norm(exit_vectors, axis=1)[:, None]

    return exit_vectors


def num_ev(m):
    """
    Get the number of exit vectors for a molecule. An exit vector is defined as the vector from an atom to a dummy atom, where it can connect to another fragment.

    :param m: RDKit molecule
    :return: int
    """
    num_evs = 0
    for b in m.GetBonds():
        if b.GetBeginAtom().GetAtomicNum() == 0 or b.GetEndAtom().GetAtomicNum() == 0:
            num_evs += 1
    return num_evs


def set_frame(coords, old_com, new_com, old_axis_angle, new_axis_angle, rotate=True):
    """
    Tranlate and rotate a fragment to a new center of mass and orientation.

    :param coords: torch tensor of shape (n, 3) where n is the number of atoms
    :param old_com: torch tensor of shape (3,) representing the old center of mass
    :param new_com: torch tensor of shape (3,) representing the new center of mass
    :param old_axis_angle: torch tensor of shape (3,) representing the old orientation in axis-angle format
    :param new_axis_angle: torch tensor of shape (3,) representing the new orientation in axis-angle format
    :param rotate: bool, whether to rotate the fragment
    :return: torch tensor of shape (n, 3) representing the new coordinates
    """
    coords = coords - old_com.unsqueeze(1)
    if rotate:
        rotmat_before = so3.matrix_from_rotation_vector(old_axis_angle)
        rotmat_after = so3.matrix_from_rotation_vector(new_axis_angle)
        rotmat_diff = rotmat_after @ rotmat_before.transpose(-1, -2)
        coords = torch.einsum('boi,bai->bao', rotmat_diff, coords)
    coords = coords + new_com.unsqueeze(1)

    return coords
