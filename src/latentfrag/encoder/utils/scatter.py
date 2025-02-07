import torch
from typing import Optional


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int,
            dim_size: Optional[int] = None, out: Optional[torch.Tensor] = None,
            reduce: Optional[str] = 'sum') -> torch.Tensor:
    """
    Mimic torch_scatter.scatter().
    :param src: Source tensor.
    :param index: Indices of elements to scatter.
    :param dim: The axis along which to index.
    :param out: Destination tensor (optional).
    :param dim_size: Size of the destination tensor if 'out' is not given.
    :param reduce: The reduce operation ("sum", "mean").
    :rtype: torch.Tensor
    """
    shape = [1] * len(src.size())
    shape[dim] = -1
    index = index.view(*shape).expand(src.size())
    # from pdb import set_trace
    # set_trace()
    # # segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))

    if out is None:
        out_shape = list(src.size())
        out_shape[dim] = dim_size
        out = src.new_full(out_shape, 0)

    out.scatter_add_(dim, index, src)

    if reduce == 'mean':
        count = src.new_zeros(out.shape)
        count.scatter_add_(dim, index, src.new_ones(src.size()))
        count[count < 1] = 1
        out = out / count

    return out
