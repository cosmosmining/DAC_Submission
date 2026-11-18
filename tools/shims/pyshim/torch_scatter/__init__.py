"""Minimal torch_scatter stand-in for machines without the compiled package (only what this code imports:
scatter_sum and scatter_softmax over dim 0 / a given dim), implemented with torch's own scatter ops."""
import torch


def _bc(index, src, dim):
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape).expand_as(src)
    return index


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    dim = dim % src.dim()
    idx = _bc(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    if out is None:
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_add(dim, idx, src)


scatter_add = scatter_sum


def scatter_max(src, index, dim=-1, dim_size=None):
    dim = dim % src.dim()
    idx = _bc(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim, idx, src, reduce="amax", include_self=True)
    return out, None


def scatter_softmax(src, index, dim=-1, dim_size=None):
    dim = dim % src.dim()
    idx = _bc(index, src, dim)
    mx, _ = scatter_max(src.detach(), index, dim, dim_size)
    shifted = src - mx.gather(dim, idx)
    ex = shifted.exp()
    den = scatter_sum(ex, index, dim, dim_size=mx.shape[dim])
    return ex / den.gather(dim, idx).clamp_min(1e-16)
