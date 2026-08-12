"""Measure throughput across MSA depths, and extrapolate to the paper's shape.

The paper reports 10.5 days for 50,000 steps on "a single H100" -- 18.1 s per
optimizer step at depth 320, crop 312, effective batch 12. That configuration
needs ~94 GB (bench/results/h100-memory-depth.json) and does not fit on the
80 GB H100 available here. The paper's own text puts their card at 96 GB, so
the number is reproducible, just not on this hardware.

Renting a bigger card does not fix the comparison, it moves the problem: an
H200 has ~4.8 TB/s of bandwidth against an H100 SXM's ~3.35, and this workload
is bandwidth-bound. A step time measured there would beat 18.1 s largely
because of the memory system, which is not the question being asked.

So this measures what does fit and extrapolates along the one axis that has to
change. Cost is close to affine in depth:

    the MSA track is O(depth * crop * d)          -- linear in depth
    the pair track is O(crop^3)                   -- constant in depth

which is the same structure the memory sweep found (0.198 GB per sequence over
a fixed 30.8 GB). `r_squared` is reported so the assumption is checked against
the data rather than asserted; below ~0.99 the extrapolation should not be
quoted. Every other knob -- crop, accumulation, effective batch, precision,
checkpointing -- is held at the paper's value, so depth is the only difference
between what was measured and what is being compared against.

An extrapolation is weaker evidence than a measurement, and this one is stated
as an estimate everywhere it appears. It is the honest option: the alternative
is a direct measurement of a different question.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import torch

from bench.config import (
    PAPER_DEPTH,
    PAPER_SECONDS_PER_STEP,
    PAPER_STEPS,
    BenchConfig,
)
from bench.train import run as train_run

# Depths to measure. The first three are measured to fit with room to spare;
# the last two probe for the ceiling and are expected to OOM on an 80 GB card,
# which is recorded rather than raised. Ordered ascending so the cheap points
# are in hand before an expensive one fails.
DEFAULT_DEPTHS: tuple[int, ...] = (64, 128, 192, 224, 256)


def fit_line(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Least-squares fit of y = slope * x + intercept, with r^2.

    Kept free of torch and of any measurement so it can be tested on known
    inputs -- an extrapolation is only as trustworthy as the fit under it, and
    a fit nobody has checked against a straight line is an assumption.
    """
    n = len(xs)
    if n < 2:
        raise ValueError(f"need at least two points to fit a line, got {n}")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all x values are identical; the fit is undefined")

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    # A perfectly flat y is fit exactly by a zero slope; r^2 is 1, not 0/0.
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def _release() -> None:
    """Drop the previous depth's model, optimizer and activations.

    Without this each successful depth leaves ~70 GB behind and every later one
    reports OOM, turning a sweep into a single measurement plus noise.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run(
    cfg: BenchConfig,
    git: dict[str, Any] | None = None,
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure `cfg` at each depth, then fit and extrapolate to the paper's.

    `cfg`'s own depth is ignored -- the sweep supplies it. Everything else is
    carried through unchanged, which is what makes the points comparable to
    each other and the extrapolation comparable to the paper.
    """
    points: list[dict[str, Any]] = []

    for depth in depths:
        _release()
        log(f"\n=== depth {depth} ===")
        try:
            result = train_run(replace_depth(cfg, depth), git=git, log=log)
        except torch.OutOfMemoryError:
            log(f"  OOM at depth {depth}")
            points.append({"depth": depth, "oom": True})
            continue
        points.append({
            "depth": depth,
            "oom": False,
            "median_step_s": result["measurements"]["median_step_s"],
            "tokens_per_s": result["measurements"]["tokens_per_s"],
            "peak_gpu_gb": result["measurements"]["peak_gpu_gb"],
            "result": result,
        })

    measured = [p for p in points if not p["oom"]]
    if not measured:
        raise RuntimeError(f"every depth OOMed: {list(depths)}. Nothing to fit.")

    fitted = len(measured) >= 2
    fit = (
        fit_line([p["depth"] for p in measured],
                 [p["median_step_s"] for p in measured])
        if fitted else None
    )

    largest = max(p["depth"] for p in measured)
    smallest_oom = min((p["depth"] for p in points if p["oom"]), default=None)

    estimate: dict[str, Any] | None = None
    if fit is not None:
        seconds = fit["slope"] * PAPER_DEPTH + fit["intercept"]
        estimate = {
            "depth": PAPER_DEPTH,
            "estimated_step_s": seconds,
            "estimated_days": seconds * PAPER_STEPS / 86_400,
            "paper_s_per_step": PAPER_SECONDS_PER_STEP,
            "ratio_vs_paper": PAPER_SECONDS_PER_STEP / seconds,
            "extrapolated_from": [p["depth"] for p in measured],
            "is_measurement": False,
        }

    return {
        "schema": 1,
        "kind": "depth-sweep",
        "config": cfg.to_dict(),
        "git": git,
        "points": points,
        "fit": fit,
        "memory_ceiling": {
            "largest_depth_measured": largest,
            "smallest_depth_that_oomed": smallest_oom,
        },
        "paper_estimate": estimate,
    }


def replace_depth(cfg: BenchConfig, depth: int) -> BenchConfig:
    """`BenchConfig` is frozen, so a sweep needs a copy per point."""
    return BenchConfig.from_dict({**cfg.to_dict(), "depth": depth})


def format_table(report: dict[str, Any]) -> str:
    lines = [f"{'depth':>7}{'s/step':>10}{'tokens/s':>12}{'peak GB':>10}"]
    for point in report["points"]:
        if point["oom"]:
            lines.append(f"{point['depth']:>7}{'OOM':>10}{'-':>12}{'-':>10}")
            continue
        peak = point["peak_gpu_gb"]
        lines.append(
            f"{point['depth']:>7}{point['median_step_s']:>10.3f}"
            f"{point['tokens_per_s']/1e3:>11.1f}k"
            f"{(f'{peak:.1f}' if peak is not None else '-'):>10}"
        )

    fit, estimate = report["fit"], report["paper_estimate"]
    if fit is not None:
        lines.append(
            f"\nfit  s/step = {fit['slope']:.5f} * depth + {fit['intercept']:.3f}"
            f"   (r^2 = {fit['r_squared']:.4f})"
        )
    if estimate is not None:
        quality = "" if fit["r_squared"] >= 0.99 else "  [r^2 below 0.99 -- do not quote]"
        lines.append(
            f"\nESTIMATE at the paper's depth {estimate['depth']} "
            f"(not measured){quality}\n"
            f"  {estimate['estimated_step_s']:.2f} s/step  ->  "
            f"{estimate['estimated_days']:.2f} days for {PAPER_STEPS:,} steps\n"
            f"  paper: {estimate['paper_s_per_step']:.1f} s/step, "
            f"{estimate['ratio_vs_paper']:.2f}x"
        )
    return "\n".join(lines)
