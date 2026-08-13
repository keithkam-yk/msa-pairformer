"""Synthetic inputs shaped exactly as the real collate emits them.

Random tokens are not a shortcut that costs accuracy here. `CollateAFBatch`
pads every batch to the fixed configured maximum -- see the commented-out
pad-to-batch-max line in `data/msa_datasets.py` -- so the tensors reaching the
model are always [B, max_depth, max_length] regardless of the MSA. Real data
changes which entries are padding; it does not change a single shape, and
therefore does not change a single FLOP.

What this does *not* stand in for is the dataloader: a3m parsing, greedy
diversity selection, and the per-example `hhfilter` subprocess. That cost is
real and deliberately excluded, and needs its own measurement.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.functional import one_hot

from msa_pairformer.features import msa_mlm, prepare_msa_masks
from msa_pairformer.tokens import nTokenTypes

BATCH_KEYS = ("msas", "msas_onehot", "mask", "msa_mask", "full_mask", "pairwise_mask")


def synthetic_batch(
    batch_size: int, depth: int, crop: int, seed: int = 0
) -> dict[str, Any]:
    """One batch with the keys and shapes `CollateAFBatch` would produce.

    Generated fully occupied -- no padding rows or columns. Occupancy affects
    only mask values, never shapes, so it cannot affect throughput; it is set
    this way for simplicity, not to flatter the numbers.
    """
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 20, (batch_size, depth, crop), generator=gen)
    masked_tokens, mlm_indices = msa_mlm(tokens, mutate_pssm=False, query_only=False)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokens)
    return {
        "msas": tokens,
        "msas_onehot": one_hot(masked_tokens, num_classes=nTokenTypes).float(),
        "masked_idx": mlm_indices,
        "mask": mask,
        "msa_mask": msa_mask,
        "full_mask": full_mask,
        "pairwise_mask": pairwise_mask,
    }


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """`masked_idx` stays on the host -- it indexes a flattened view and is
    consumed by advanced indexing, which accepts a CPU index tensor."""
    return {
        **batch,
        **{key: batch[key].to(device) for key in BATCH_KEYS},
    }
