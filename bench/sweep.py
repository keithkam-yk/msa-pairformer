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

# Depths to measure. Six fitting points below the 80 GB ceiling, then 224 to
# confirm where it is. Ordered ascending so the cheap points are in hand before
# an expensive one fails.
#
# The first sweep used (64, 128, 192, 224, 256) and got three fitting points,
# because 224 and 256 OOM. Three points against two parameters leaves one
# residual degree of freedom: adjusted r^2 is undefined, and the whole
# departure from linearity collapses to a single number -- how far the middle
# point falls from the chord of the outer two. It reported r^2 = 0.9945 and
# meant almost nothing by it. The ceiling caps depth, but nothing capped how
# many points fit underneath it.
DEFAULT_DEPTHS: tuple[int, ...] = (32, 64, 96, 128, 160, 192, 224)

# Below this many measured points, an extrapolation is not reported as
# quotable, whatever r^2 says. Two parameters plus a residual to check them
# against is the bare minimum; four leaves two.
MIN_POINTS_TO_QUOTE = 5


def fit_line(xs: list[float], ys: list[float]) -> dict[str, Any]:
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

    # Reported alongside r^2 because r^2 alone cannot be read without it. At
    # n=3 against two parameters there is one residual degree of freedom and
    # adjusted r^2 is undefined -- a high r^2 there says the arithmetic worked,
    # not that the relationship is linear. None is the honest value.
    dof = n - 2
    adjusted = (
        1.0 - (1.0 - r_squared) * (n - 1) / (dof - 1) if dof > 1 else None
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted,
        "n_points": n,
        "residual_dof": dof,
    }


def curvature_bracket(xs: list[float], ys: list[float], at: float) -> dict[str, Any]:
    """Bracket an extrapolation between a straight line and a parabola.

    r^2 measures fit over the range that was measured. An extrapolation lives
    outside it, where the thing that matters is curvature -- and mild curvature
    is exactly what a high r^2 hides. Fitting a quadratic to the same points
    and extrapolating both gives the spread the data cannot distinguish
    between.

    The quadratic is not the better model; at n=3 it fits three points exactly
    and is guaranteed to. It is an alternative the measurements are equally
    consistent with, and the gap between the two is the part of the estimate
    that is not evidence.
    """
    import numpy as np

    line = fit_line(xs, ys)
    linear = line["slope"] * at + line["intercept"]

    quadratic: float | None = None
    if len(xs) >= 3:
        c2, c1, c0 = (float(c) for c in np.polyfit(np.array(xs), np.array(ys), 2))
        quadratic = c2 * at * at + c1 * at + c0

    values = [v for v in (linear, quadratic) if v is not None]
    return {
        "at": at,
        "linear": linear,
        "quadratic": quadratic,
        "low": min(values),
        "high": max(values),
        # How much of the estimate is model choice rather than measurement.
        "spread_fraction": (max(values) - min(values)) / min(values),
    }


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
        depths_measured = [float(p["depth"]) for p in measured]
        times = [float(p["median_step_s"]) for p in measured]
        bracket = curvature_bracket(depths_measured, times, float(PAPER_DEPTH))
        seconds = bracket["linear"]
        # Both conditions, because either alone is satisfiable while the
        # estimate is still worthless: r^2 is near 1 on any three points, and a
        # large n does not rescue a relationship that is not a line.
        quotable = (
            len(measured) >= MIN_POINTS_TO_QUOTE and fit["r_squared"] >= 0.99
        )
        estimate = {
            "depth": PAPER_DEPTH,
            "estimated_step_s": seconds,
            "estimated_step_s_low": bracket["low"],
            "estimated_step_s_high": bracket["high"],
            "estimated_days": seconds * PAPER_STEPS / 86_400,
            "paper_s_per_step": PAPER_SECONDS_PER_STEP,
            "ratio_vs_paper": PAPER_SECONDS_PER_STEP / seconds,
            "ratio_vs_paper_low": PAPER_SECONDS_PER_STEP / bracket["high"],
            "ratio_vs_paper_high": PAPER_SECONDS_PER_STEP / bracket["low"],
            "curvature": bracket,
            "extrapolated_from": [p["depth"] for p in measured],
            "extrapolation_reach": PAPER_DEPTH / largest,
            "is_measurement": False,
            "quotable": quotable,
            "min_points_to_quote": MIN_POINTS_TO_QUOTE,
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
        adj = fit["adjusted_r_squared"]
        adj_text = f"{adj:.4f}" if adj is not None else "undefined"
        lines.append(
            f"\nfit  s/step = {fit['slope']:.5f} * depth + {fit['intercept']:.3f}"
            f"\n  n = {fit['n_points']} points, residual dof = {fit['residual_dof']}"
            f", r^2 = {fit['r_squared']:.4f}, adjusted r^2 = {adj_text}"
        )
    if estimate is not None:
        reasons = []
        if len(estimate["extrapolated_from"]) < estimate["min_points_to_quote"]:
            reasons.append(
                f"only {len(estimate['extrapolated_from'])} points, "
                f"need {estimate['min_points_to_quote']}"
            )
        if fit["r_squared"] < 0.99:
            reasons.append(f"r^2 {fit['r_squared']:.4f} below 0.99")
        verdict = "" if estimate["quotable"] else f"  [DO NOT QUOTE: {'; '.join(reasons)}]"

        curve = estimate["curvature"]
        lines.append(
            f"\nESTIMATE at the paper's depth {estimate['depth']} "
            f"(not measured){verdict}\n"
            f"  {estimate['estimated_step_s']:.2f} s/step linear, "
            f"range {curve['low']:.2f}-{curve['high']:.2f} s "
            f"({curve['spread_fraction']*100:.0f}% model choice, not measurement)\n"
            f"  {estimate['estimated_days']:.2f} days for {PAPER_STEPS:,} steps\n"
            f"  paper: {estimate['paper_s_per_step']:.1f} s/step, "
            f"{estimate['ratio_vs_paper_low']:.2f}-"
            f"{estimate['ratio_vs_paper_high']:.2f}x\n"
            f"  extrapolating {estimate['extrapolation_reach']:.2f}x beyond the "
            f"largest measured depth"
        )
    return "\n".join(lines)
