import random
import warnings
from typing import Union, Iterable
from argparse import Namespace
from contextlib import contextmanager

import torch
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import KekulizeException, AtomKekulizeException


class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def bvm(v, m):
    """
    Batched vector-matrix product of the form out = v @ m
    :param v: (b, n_in)
    :param m: (b, n_in, n_out)
    :return: (b, n_out)
    """
    # return (v.unsqueeze(1) @ m).squeeze()
    # if len(v.shape) == 1:
    #     v = v.unsqueeze(0)
    return torch.bmm(v.unsqueeze(1), m).squeeze(1)


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_rdkit_logging():
    # RDLogger.DisableLog('rdApp.*')
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')


def write_sdf_file(sdf_path, molecules, catch_errors=True):
    with Chem.SDWriter(str(sdf_path)) as w:
        for mol in molecules:
            try:
                if mol is None:
                    raise ValueError("Mol is None.")
                w.write(mol)

            except (RuntimeError, ValueError) as e:
                if not catch_errors:
                    raise e

                if isinstance(e, (KekulizeException, AtomKekulizeException)):
                    w.SetKekulize(False)
                    w.write(mol)
                    w.SetKekulize(True)
                    warnings.warn(f"Mol saved without kekulization.")
                else:
                    # write empty mol to preserve the original order
                    w.write(Chem.Mol())
                    warnings.warn(f"Erroneous mol replaced with empty dummy.")


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)


def batch_to_list(data, batch_mask, keep_order=True):
    if keep_order:  # preserve order of elements within each sample
        data_list = [data[batch_mask == i]
                     for i in torch.unique(batch_mask, sorted=True)]
        return data_list

    # make sure batch_mask is increasing
    idx = torch.argsort(batch_mask)
    batch_mask = batch_mask[idx]
    data = data[idx]

    chunk_sizes = torch.unique(batch_mask, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


def batch_to_list_for_indices(indices, batch_mask, offsets=None):
    # (2, n) -> (n, 2)
    split = batch_to_list(indices.T, batch_mask)

    # rebase indices at zero & (n, 2) -> (2, n)
    if offsets is None:
        warnings.warn("Trying to infer index offset from smallest element in "
                      "batch. This might be wrong.")
        split = [x.T - x.min() for x in split]
    else:
        # Typically 'offsets' would be accumulate(sizes[:-1], initial=0)
        assert len(offsets) == len(split)
        split = [x.T - offset for x, offset in zip(split, offsets)]

    return split


def get_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        norm_type: float = 2.0) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack(
        [torch.norm(p.grad.detach(), norm_type).to(device) for p in
         parameters]), norm_type)

    return total_norm


def pointcloud_to_pdb(filename, coords, feature=None, show_as='HETATM'):

    assert len(coords) <= 99999
    assert show_as in {'ATOM', 'HETATM'}
    if feature is not None:
        assert feature.min() >= 0 and feature.max() <= 100

    out = ""
    for i in range(len(coords)):
        # format:
        # 'ATOM'/'HETATM' <atom serial number> <atom name>
        # <alternate location indicator> <residue name> <chain>
        # <residue sequence number> <code for insertions of residues>
        # <x> <y> <z> <occupancy> <b-factor> <segment identifier>
        # <element symbol> <charge>
        feature_i = 0.0 if feature is None else feature[i]
        out += f"{show_as:<6}{i:>5} {'SURF':<4}{' '}{'ABC':>3} {'A':>1}{1:>4}" \
               f"{' '}   {coords[i, 0]:>8.3f}{coords[i, 1]:>8.3f}" \
               f"{coords[i, 2]:>8.3f}{1.0:>6}{feature_i:>6.2f}      {'':>4}" \
               f"{'H':>2}{'':>2}\n"

    with open(filename, 'w') as f:
        f.write(out)


def process_in_chunks(adj_pocket, chunk_size=1000):
    n = adj_pocket.size(0)
    edges_list = []

    for i in range(0, n, chunk_size):
        chunk = adj_pocket[i:i+chunk_size]
        chunk_edges = torch.nonzero(chunk)
        chunk_edges[:, 0] = chunk_edges[:, 0] + i  # Adjust the row indices
        edges_list.append(chunk_edges)

    return torch.cat(edges_list, dim=0).t()


def dict_to_namespace(input_dict):
    """ Recursively convert a nested dictionary into a Namespace object """
    if isinstance(input_dict, dict):
        output_namespace = Namespace()
        output = output_namespace.__dict__
        for key, value in input_dict.items():
            output[key] = dict_to_namespace(value)
        return output_namespace

    elif isinstance(input_dict, Namespace):
        # recurse as Namespace might contain dictionaries
        return dict_to_namespace(input_dict.__dict__)

    else:
        return input_dict


def namespace_to_dict(x):
    """ Recursively convert a nested Namespace object into a dictionary. """
    if not (isinstance(x, Namespace) or isinstance(x, dict)):
        return x

    if isinstance(x, Namespace):
        x = vars(x)

    # recurse
    output = {}
    for key, value in x.items():
        output[key] = namespace_to_dict(value)
    return output


@contextmanager
def temp_seed(seed):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)