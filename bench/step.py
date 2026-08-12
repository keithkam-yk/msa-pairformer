"""Model construction and a single forward + backward.

This is where the harness has to disagree with the library's defaults, so it is
kept apart from the timing loop that calls it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from msa_pairformer import pairwise_operations
from msa_pairformer.model import MSAPairformer

# Snapshot before anything can override it: `triangle_path` must never claim
# cuEquivariance on a host that does not have it.
CUEQUIVARIANCE_PRESENT = pairwise_operations.CUEQUIVARIANCE_AVAILABLE


@contextmanager
def triangle_path(use_cuequivariance: bool) -> Iterator[bool]:
    """Choose the triangle implementation for modules built inside the block.

    `PairwiseBlock` and `TriangleMultiplication` read the module-level
    `CUEQUIVARIANCE_AVAILABLE` directly at construction
    (pairwise_operations.py:451) rather than taking it as an argument, so
    overriding that constant is the only way to select the path.

    It restores the previous value on exit. A one-way setter would leak: two
    variants measured in a single process would silently share whichever path
    the first one selected, and every comparison after it would be worthless.

    Yields whether cuEquivariance is actually in effect, which is not the same
    as what was asked for -- requesting it on a host without it yields False.
    """
    previous = pairwise_operations.CUEQUIVARIANCE_AVAILABLE
    effective = bool(use_cuequivariance and CUEQUIVARIANCE_PRESENT)
    pairwise_operations.CUEQUIVARIANCE_AVAILABLE = effective
    try:
        yield effective
    finally:
        pairwise_operations.CUEQUIVARIANCE_AVAILABLE = previous


def build_model(
    device: torch.device, use_cuequivariance: bool = True, seed: int = 0
) -> MSAPairformer:
    """Build with a fixed seed so both triangle paths get identical weights --
    without that, any comparison between them measures initialisation too."""
    with triangle_path(use_cuequivariance):
        torch.manual_seed(seed)
        model = MSAPairformer().to(device)
    model.train()
    return model


def micro_step(
    model: MSAPairformer,
    batch: dict[str, Any],
    criterion: torch.nn.Module,
    accum: int,
) -> torch.Tensor:
    """One forward + backward. Returns the unscaled loss for reporting.

    Four of `forward`'s defaults are wrong for training and all four cost time:

    * `return_cb_contacts` / `return_confind_contacts` default to True and run
      the contact heads, which MLM training never uses.
    * `store_msa_repr_cpu` defaults to True, which with `query_only=False`
      sends the final [b, s, n, d] representation to the host and straight back
      (model.py:258 and :520) -- a synchronising round trip on the autograd
      path, paid once forward and again in reverse through the backward.
    * `store_pairwise_repr_cpu` is the same trap on the pair track.
    """
    results = model(
        msa=batch["msas_onehot"],
        mask=batch["mask"],
        msa_mask=batch["msa_mask"],
        full_mask=batch["full_mask"],
        pairwise_mask=batch["pairwise_mask"],
        query_only=False,
        return_cb_contacts=False,
        return_confind_contacts=False,
        store_msa_repr_cpu=False,
        store_pairwise_repr_cpu=False,
    )
    logits = results["logits"]
    pred = logits.view(-1, logits.shape[-1])[batch["masked_idx"]]
    target = batch["msas"].view(-1)[batch["masked_idx"]]
    loss = criterion(pred.float(), target)
    # Scale so accumulated gradients average rather than sum.
    (loss / accum).backward()
    return loss.detach()
