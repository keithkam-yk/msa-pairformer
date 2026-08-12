"""The cheap gate on `torch.compile`, before any depth ladder is rented.

A full sweep is four variants times seven depths of 111M-parameter builds. Two
questions decide whether that is worth paying for, and both can be answered at
a shape small enough to be nearly free:

1. **Does it run at all?** `PairwiseBlock` wraps both triangle updates in
   `torch.utils.checkpoint(..., use_reentrant=False)` and checkpointing is not
   optional at these shapes. Compilation across a checkpoint boundary is the
   riskiest interaction here.

2. **What actually compiled?** On the cuEquivariance path the fused triangle
   kernels are opaque to Dynamo, so a graph break at each call is expected and
   the compiled region is the elementwise work between them. That is a
   different claim from "compile made the model faster", and the graph-break
   reasons are what tell the two apart.

It also answers a third question that is not about speed at all: whether the
compiled path computes the same thing. The golden fixtures in
`tests/fixtures/golden.pt` were recorded eager and per module, so they say
nothing about a compiled whole model under checkpointing. This compares loss
and gradient norm between each compiled variant and its own eager counterpart,
at identical weights on an identical batch, which is the comparison that
matters for a training run.

Peak memory is reported for the same reason. Compilation changes which
activations survive, and a compiled variant whose ceiling moved *down* would
show up in a sweep as "compile OOMs earlier" -- read as a limitation of
compilation when it is really a recompute-policy interaction. Better to see it
here.

    python bench/probe.py --device cpu --depth 4 --crop 8
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from torch.nn import CrossEntropyLoss

from bench.data import synthetic_batch, to_device
from bench.drift import set_float32_precision
from bench.provenance import environment, git_info
from bench.step import (
    CUEQUIVARIANCE_PRESENT,
    build_model,
    compile_stats,
    compile_targets,
    micro_step,
)

Logger = Callable[[str], None]

# (label, cuequivariance, compile_mode). Each compiled entry is compared
# against the eager entry immediately above it, so order is load-bearing.
PROBE_VARIANTS: tuple[tuple[str, bool, str], ...] = (
    ("vanilla", False, "off"),
    ("vanilla+compile", False, "default"),
    ("cuequivariance", True, "off"),
    ("cuequivariance+compile", True, "default"),
)


def probe_one(
    device: torch.device,
    use_cuequivariance: bool,
    compile_mode: str,
    depth: int,
    crop: int,
    batch: dict[str, Any],
    log: Logger = print,
) -> dict[str, Any]:
    """Two forward+backward passes, and everything they reveal.

    Two, not one: the first pays for compilation and the second does not, so
    the pair separates the one-off cost from the steady-state cost. Neither is
    a throughput measurement -- the shape is far too small and the optimizer
    never runs -- and neither is reported as one.

    Gradients are read back as a single norm rather than compared elementwise.
    The question is whether the compiled path trains the same model, and a
    norm over all 111M parameters answers it while staying one number wide.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(
        device, use_cuequivariance=use_cuequivariance, compile_mode=compile_mode
    )
    criterion = CrossEntropyLoss()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    autocast_on = device.type == "cuda"
    times: list[float] = []
    loss_value = float("nan")
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        sync()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_on):
            loss = micro_step(model, batch, criterion, 1, checkpoint_triangles=True)
        sync()
        times.append(time.perf_counter() - started)
        loss_value = loss.item()

    grad_norm = torch.norm(
        torch.stack([
            p.grad.detach().float().norm()
            for p in model.parameters() if p.grad is not None
        ])
    ).item()

    stats = compile_stats() if compile_mode != "off" else None
    return {
        "cuequivariance": use_cuequivariance,
        "compile_mode": compile_mode,
        "compiled_modules": len(compile_targets(model)),
        "first_pass_s": times[0],
        "second_pass_s": times[1],
        "loss": loss_value,
        "grad_norm": grad_norm,
        "peak_gpu_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda" else None
        ),
        "compile": stats,
    }


def compare(eager: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
    """How far the compiled path moved from its own eager counterpart.

    Relative, because the absolute scale of a cross-entropy loss and of a
    gradient norm over 111M parameters carry no shared meaning. bf16 autocast
    alone puts a floor of roughly 1e-2 relative under any such comparison, so
    a small number here is agreement, not identity -- and identity is not what
    a different kernel schedule can deliver.
    """
    def relative(key: str) -> float:
        want, got = eager[key], compiled[key]
        return abs(got - want) / max(abs(want), 1e-12)

    return {
        "loss_rel": relative("loss"),
        "grad_norm_rel": relative("grad_norm"),
        "speedup_second_pass": eager["second_pass_s"] / compiled["second_pass_s"],
        "compile_overhead_s": compiled["first_pass_s"] - compiled["second_pass_s"],
        "peak_gb_delta": (
            compiled["peak_gpu_gb"] - eager["peak_gpu_gb"]
            if eager["peak_gpu_gb"] is not None else None
        ),
    }


def run(
    device: torch.device | None = None,
    depth: int = 32,
    crop: int = 64,
    log: Logger = print,
) -> dict[str, Any]:
    """Probe every variant this host can run, on one shared batch."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    precision = set_float32_precision("highest")
    batch = to_device(synthetic_batch(1, depth, crop), device)

    results: dict[str, Any] = {}
    for label, use_cuex, compile_mode in PROBE_VARIANTS:
        if use_cuex and not CUEQUIVARIANCE_PRESENT:
            log(f"skipping {label}: cuEquivariance is not available here")
            continue
        log(f"\n=== {label} ===")
        results[label] = probe_one(
            device, use_cuex, compile_mode, depth, crop, batch, log=log
        )

    comparisons = {
        label: compare(results[label.removesuffix("+compile")], results[label])
        for label in results
        if label.endswith("+compile") and label.removesuffix("+compile") in results
    }

    return {
        "schema": 1,
        "kind": "compile-probe",
        "git": git_info(),
        "environment": environment(),
        "precision": precision,
        "shape": {"micro_batch": 1, "depth": depth, "crop": crop},
        "cuequivariance_present": CUEQUIVARIANCE_PRESENT,
        "variants": results,
        "comparisons": comparisons,
    }


def format_table(report: dict[str, Any]) -> str:
    shape = report["shape"]
    lines = [
        f"probe shape [1, {shape['depth']}, {shape['crop']}] -- not a throughput "
        "measurement",
        f"\n{'variant':<24}{'1st pass s':>12}{'2nd pass s':>12}{'peak GB':>10}"
        f"{'graphs':>9}{'breaks':>8}",
    ]
    for label, entry in report["variants"].items():
        stats = entry["compile"] or {}
        peak = entry["peak_gpu_gb"]
        lines.append(
            f"{label:<24}{entry['first_pass_s']:>12.2f}"
            f"{entry['second_pass_s']:>12.2f}"
            f"{(f'{peak:.2f}' if peak is not None else '-'):>10}"
            f"{stats.get('unique_graphs', '-'):>9}"
            f"{stats.get('graph_breaks', '-'):>8}"
        )

    for label, entry in report["variants"].items():
        reasons = (entry["compile"] or {}).get("graph_break_reasons")
        if reasons:
            lines.append(f"\ngraph breaks under {label}")
            for reason, count in reasons.items():
                lines.append(f"  {count:>4}  {reason}")

    for label, delta in report["comparisons"].items():
        gb = delta["peak_gb_delta"]
        lines.append(
            f"\n{label} against its eager counterpart"
            f"\n  loss differs by {delta['loss_rel']:.2e} relative, "
            f"gradient norm by {delta['grad_norm_rel']:.2e}"
            f"\n  second pass {delta['speedup_second_pass']:.2f}x, "
            f"first pass cost {delta['compile_overhead_s']:.1f} s more"
            + (f"\n  peak memory {gb:+.2f} GB" if gb is not None else "")
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    report = run(
        device=torch.device(args.device) if args.device else None,
        depth=args.depth,
        crop=args.crop,
    )
    print("\n" + format_table(report))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
