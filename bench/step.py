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
from msa_pairformer.core import PreLayerNorm, Transition
from msa_pairformer.model import MSAPairformer
from msa_pairformer.outer_product import OuterProduct
from msa_pairformer.pairwise_operations import MSAPairWeightedAveraging, PairwiseBlock

# Snapshot before anything can override it: `triangle_path` must never claim
# cuEquivariance on a host that does not have it.
CUEQUIVARIANCE_PRESENT = pairwise_operations.CUEQUIVARIANCE_AVAILABLE

# What `torch.compile` is pointed at. These are the four units `CoreModule`
# repeats 22 times, so one compilation is reused by every layer.
#
# Compiling `MSAPairformer` itself is the obvious alternative and it is the
# wrong one. Its forward is a Python loop over `self.layers` carrying dict
# accumulation, `del`, `enumerate`, `in` tests against index lists and a
# conditional `break` (model.py:180-232). Dynamo would unroll all 22 layers
# into one graph and guard on all of that -- a long compile of a graph that is
# rebuilt whenever an unrelated argument changes. The repeated leaves have none
# of it, and they hold the arithmetic.
#
# `PreLayerNorm` is listed ahead of the `Transition` it wraps so the norm joins
# the graph rather than sitting outside it. A LayerNorm followed by a linear is
# exactly the pair Inductor fuses; splitting them at a compile boundary would
# throw away the easiest win available.
COMPILE_TARGETS = (
    MSAPairWeightedAveraging,
    OuterProduct,
    PairwiseBlock,
    PreLayerNorm,
    Transition,
)


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


def compile_targets(model: torch.nn.Module) -> list[str]:
    """Names of the modules `compile_in_place` would compile.

    Separated from the compilation so the selection can be checked on a CPU
    without a GPU or an Inductor run.

    A target inside another target is skipped. `PairwiseBlock` contains a
    `Transition`, and both are in `COMPILE_TARGETS`; compiling the inner one
    too would put a `torch.compile` boundary in the middle of a region Dynamo
    is already tracing, which at best wastes a compilation and at worst splits
    a graph that did not need splitting. `named_modules` yields parents before
    children, so the first match along a path wins.
    """
    chosen: list[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(f"{parent}.") for parent in chosen):
            continue
        if isinstance(module, COMPILE_TARGETS):
            chosen.append(name)
    return chosen


def compile_in_place(model: torch.nn.Module, mode: str) -> list[str]:
    """Compile the repeated leaves, and return which ones were compiled.

    `Module.compile` rather than `torch.compile(module)`: the former swaps the
    module's call implementation and leaves the module itself in place, so
    parameter names, `state_dict` keys and the module tree are unchanged. The
    latter returns an `OptimizedModule` wrapper that prefixes every key with
    `_orig_mod.`, which would make a compiled run's weights a different shape
    of object from every other run's for no benefit here.

    `dynamic` is left at its default. Within one `train.run` the shapes never
    change, so the first compilation is static and specialised, which is what
    should be measured. A sweep builds a fresh model for each depth, and the
    pair track does not carry the depth axis at all, so most of the work is
    reused across depths rather than repeated.
    """
    names = compile_targets(model)
    for name in names:
        model.get_submodule(name).compile(mode=mode)
    return names


def compile_stats() -> dict[str, Any]:
    """How much of the model Dynamo actually captured.

    The number that matters is `unique_graphs`. If compilation silently fell
    back to eager -- a suppressed Dynamo error, a mode that did nothing -- this
    is zero and the run would otherwise report the eager path under the
    compiled label. That is the same failure `train.run` already refuses for
    cuEquivariance, and it is likelier here.

    `graph_breaks` is the interesting half. On the cuEquivariance path the
    fused triangle kernels are opaque to Dynamo, so a break at each call is
    expected and the compiled region is the elementwise work between them; the
    reasons say whether that is what happened.
    """
    from torch._dynamo.utils import counters

    breaks = counters.get("graph_break", {})
    return {
        "unique_graphs": counters.get("stats", {}).get("unique_graphs", 0),
        "graph_breaks": sum(breaks.values()),
        # Ten is enough to see the pattern; the tail is the same reasons again.
        "graph_break_reasons": dict(
            sorted(breaks.items(), key=lambda kv: -kv[1])[:10]
        ),
    }


def reset_compile_stats() -> None:
    """Zero the counters, so `compile_stats` describes this model only.

    Dynamo's counters are process-global and a sweep measures many models in
    one process. Without this, depth 224 would report every graph break since
    depth 32.
    """
    from torch._dynamo.utils import counters

    counters.clear()


def build_model(
    device: torch.device,
    use_cuequivariance: bool = True,
    seed: int = 0,
    compile_mode: str = "off",
) -> MSAPairformer:
    """Build with a fixed seed so every variant gets identical weights --
    without that, any comparison between them measures initialisation too.

    Compilation happens here rather than around the step function so the
    backward stays eager and the loss stays out of the graph. Nothing about
    `micro_step` changes when this is on.
    """
    with triangle_path(use_cuequivariance):
        torch.manual_seed(seed)
        model = MSAPairformer().to(device)
    model.train()
    if compile_mode != "off":
        reset_compile_stats()
        compile_in_place(model, compile_mode)
    return model


def micro_step(
    model: MSAPairformer,
    batch: dict[str, Any],
    criterion: torch.nn.Module,
    accum: int,
    checkpoint_triangles: bool = True,
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

    `use_checkpointing_triangles` is the fifth default that is wrong for
    training, and it is wrong differently: the other four cost time, this one
    decides whether the paper's configuration runs at all. Without it, depth
    320 at a 312 crop allocates 76.2 GiB and dies on an 80 GB H100, with
    micro_batch already at 1. It trades recompute for memory, so it makes the
    measured step time slower -- and a number that exists beats one that OOMs.
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
        use_checkpointing_triangles=checkpoint_triangles,
    )
    logits = results["logits"]
    pred = logits.view(-1, logits.shape[-1])[batch["masked_idx"]]
    target = batch["msas"].view(-1)[batch["masked_idx"]]
    loss = criterion(pred.float(), target)
    # Scale so accumulated gradients average rather than sum.
    (loss / accum).backward()
    return loss.detach()
