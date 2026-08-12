# Adapted from https://github.com/yoakiyama/openfold/blob/main/openfold/utils/chunk_utils.py

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict

def spec_dict_map(fn_d, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn_d[k], v, leaf_type)
        else:
            new_dict[k] = tree_map(fn_d[k], v, leaf_type)

    return new_dict

def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict) and isinstance(fn, dict):
        return spec_dict_map(fn, tree, leaf_type)
    elif isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")
tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)

def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")

    return shapes

def _flat_idx_to_idx(flat_idx: int, dims: Sequence[int]) -> tuple[int, ...]:
    """Convert an index into the flattened batch dimensions into a per-dimension index."""
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d

    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: Sequence[int],
    start_edges: Sequence[bool] | None = None,
    end_edges: Sequence[bool] | None = None,
) -> list[tuple[slice, ...]]:
    """Produce an ordered sequence of tensor slices that, applied to a tensor with shape
    ``dims``, together yield every leaf in the contiguous range [start, end].

    ``end`` is INCLUSIVE. The sequence of slices is kept short so that the caller performs
    as few indexing operations as possible.
    """
    # start_edges/end_edges indicate whether, starting from a given dimension, the
    # start/end index sits on the top/bottom edge of the corresponding subtree.
    def reduce_edge_list(edges: list[bool]) -> None:
        tally = True
        for i in range(len(edges)):
            reversed_idx = -1 * (i + 1)
            edges[reversed_idx] = edges[reversed_idx] and tally
            tally = edges[reversed_idx]

    if start_edges is None:
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [e == (d - 1) for e, d in zip(end, dims, strict=False)]
        reduce_edge_list(end_edges)

    # Base cases: either there is nothing left to slice, or the remaining
    # one-dimensional range can be sliced directly.
    if len(start) == 0:
        return [()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    slices: list[tuple[slice, ...]] = []
    path_l: list[slice] = []

    # Dimensions in which start and end agree can be selected directly
    for s, e in zip(start, end, strict=False):
        if s == e:
            path_l.append(slice(s, s + 1))
        else:
            break

    path = tuple(path_l)
    divergence_idx = len(path)

    # start == end, and we're done
    if divergence_idx == len(dims):
        return [path]

    def upper() -> list[tuple[slice, ...]]:
        sdi = start[divergence_idx]
        return [
            (*path, slice(sdi, sdi + 1), *s)
            for s in _get_minimal_slice_set(
                start[divergence_idx + 1:],
                [d - 1 for d in dims[divergence_idx + 1:]],
                dims[divergence_idx + 1:],
                start_edges=start_edges[divergence_idx + 1:],
                end_edges=[True for _ in end_edges[divergence_idx + 1:]],
            )
        ]

    def lower() -> list[tuple[slice, ...]]:
        edi = end[divergence_idx]
        return [
            (*path, slice(edi, edi + 1), *s)
            for s in _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1:]],
                end[divergence_idx + 1:],
                dims[divergence_idx + 1:],
                start_edges=[True for _ in start_edges[divergence_idx + 1:]],
                end_edges=end_edges[divergence_idx + 1:],
            )
        ]

    # If both start and end are at the edges of the subtree rooted at divergence_idx,
    # the whole subtree can be selected at once
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append((*path, slice(start[divergence_idx], end[divergence_idx] + 1)))
    # If only start is at the edge, grab almost all of the subtree and treat the ragged
    # bottom edge as a special case
    elif start_edges[divergence_idx]:
        slices.append((*path, slice(start[divergence_idx], end[divergence_idx])))
        slices.extend(lower())
    # As above, but the top is the ragged one this time
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append((*path, slice(start[divergence_idx] + 1, end[divergence_idx] + 1)))
    # Both sides are ragged: handle each separately, and take whatever contiguous
    # ground lies between them in one chunk
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append((*path, slice(start[divergence_idx] + 1, end[divergence_idx])))
        slices.extend(lower())

    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(
    t: torch.Tensor,
    flat_start: int,
    flat_end: int,
    no_batch_dims: int,
) -> torch.Tensor:
    """Equivalent to

        t.reshape((-1, *t.shape[no_batch_dims:]))[flat_start:flat_end]

    but without the up-front reshape, which materialises the whole (possibly expanded)
    tensor. Only sub-tensors that scale with the chunk size (flat_end - flat_start) are
    ever reshaped here.
    """
    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # _get_minimal_slice_set is inclusive
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # Get an ordered list of slices to perform
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    sliced_tensors = [t[s] for s in slices]

    return torch.cat([s.reshape((-1, *t.shape[no_batch_dims:])) for s in sliced_tensors])


def chunk_layer(
    layer: Callable,
    inputs: dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
    select_chunk_fn_d: dict[str, Callable] | None = None,
    orig_batch_dims: dict[str, tuple] | None = None,
    flat_batch_dim: int | None = None,
    og_batch_dim: tuple | None = None,
) -> Any:
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    if orig_batch_dims is None:
        orig_batch_dims = {}
        initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
        for k in inputs.keys():
            orig_batch_dims[k] = tuple([max(s) for s in zip(*initial_dims, strict=False)])
        og_batch_dim = tuple([max(s) for s in zip(*initial_dims, strict=False)])

    def _prep_inputs(t, batch_dim):
            if(not low_mem):
                if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                    t = t.expand(batch_dim + t.shape[no_batch_dims:])
                t = t.reshape(-1, *t.shape[no_batch_dims:])
            else:
                t = t.expand(batch_dim + t.shape[no_batch_dims:])
            return t
    prep_inputs_fn_d = {}
    for k in inputs.keys():
        prep_inputs_fn_d[k] = partial(_prep_inputs, batch_dim=orig_batch_dims[k])

    prepped_inputs = tensor_tree_map(prep_inputs_fn_d, inputs)
    prepped_outputs = None
    if(_out is not None):
        reshape_fn = lambda t: t.view([-1, *list(t.shape[no_batch_dims:])])
        prepped_outputs = tensor_tree_map(reshape_fn, _out)

    if flat_batch_dim is None:
        flat_batch_dim = 1
        for d in orig_batch_dims:
            flat_batch_dim *= orig_batch_dims[d][0]

    no_chunks = flat_batch_dim // chunk_size + (
        flat_batch_dim % chunk_size != 0
    )

    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        # Chunk the input
        if(not low_mem):
            if select_chunk_fn_d is None:
                select_chunk = (
                    lambda t: t[i : i + chunk_size] if t.shape[0] != 1 else t
                )
            else:
                select_chunk = {k: partial(select_chunk_fn_d[k], i=i, chunk_size=chunk_size) for k in select_chunk_fn_d.keys()}
        else:
            select_chunk = (
                partial(
                    _chunk_slice, 
                    flat_start=i, 
                    flat_end=min(flat_batch_dim, i + chunk_size),
                    no_batch_dims=no_batch_dims
                )
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim, *t.shape[1:]))
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:
            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        if(_add_into_out):
                            v[i: i + chunk_size] += d2[k]
                        else:
                            v[i: i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk, strict=False):
                if(_add_into_out):
                    x1[i: i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            if(_add_into_out):
                out[i: i + chunk_size] += output_chunk
            else:
                out[i: i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.view(og_batch_dim + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out