"""Minimal training-throughput harness for MSA Pairformer.

The published repository has no training loop -- no backward pass, no optimizer
step anywhere outside `pairing_optimization/`. This reconstructs the smallest
one faithful enough to measure against: masked-language-model loss over the full
MSA, bf16 autocast, gradient accumulation, AdamW.

Defaults reproduce the configuration reported in the paper (see
`bench.config`). The reported run was 10.5 days on a single H100 across two
phases -- 1,176,000 alignments, 771 ms each -- so this times a handful of steps
and extrapolates to that horizon for comparison.

This module must never import `modal`. Staying device-agnostic is what lets the
identical code run on a laptop CPU, on a rented H100, and later on several GPUs
without edits, and it is what makes the CPU self-check a real guard on the code
that runs remotely. `bench/modal_app.py` is a thin adapter around `run()`.

    python bench/train.py
    python bench/train.py --device cpu --depth 8 --crop 12 --steps 2 --accum 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, get_args

import torch
from torch.nn import CrossEntropyLoss

from bench.config import (
    PAPER_SECONDS_PER_EXAMPLE,
    REPORTED_DAYS,
    TOTAL_EXAMPLES,
    BenchConfig,
    CompileMode,
)
from bench.data import synthetic_batch, to_device
from bench.drift import set_float32_precision
from bench.provenance import environment, git_info
from bench.step import (
    CUEQUIVARIANCE_PRESENT,
    build_model,
    compile_stats,
    micro_step,
)

Logger = Callable[[str], None]


def run(
    cfg: BenchConfig | dict[str, Any],
    git: dict[str, Any] | None = None,
    log: Logger = print,
) -> dict[str, Any]:
    """Measure one configuration and return a self-describing result.

    `git` lets a caller inject repository state captured elsewhere, since the
    Modal image has no `.git` of its own.
    """
    if isinstance(cfg, dict):
        cfg = BenchConfig.from_dict(cfg)

    # Fail before anything is billed. A container where the cuequivariance
    # wheels did not install would otherwise run vanilla under the
    # `cuequivariance` label and report two identical baselines as a finding.
    if cfg.cuequivariance and not CUEQUIVARIANCE_PRESENT:
        raise RuntimeError(
            "config requests cuEquivariance but it is not available on this host "
            "(needs linux + CUDA + cuequivariance_torch). Refusing to silently "
            "measure the vanilla path under the cuEquivariance label; pass "
            "cuequivariance=False if the fallback is what you meant to measure."
        )

    device = torch.device(cfg.device)
    autocast_on = cfg.amp == "bf16" and device.type == "cuda"
    precision = set_float32_precision(cfg.float32_precision)

    # Before build_model, which may switch the triangle path: this field
    # describes the machine, not the variant.
    env = environment()

    model = build_model(
        device,
        use_cuequivariance=cfg.cuequivariance,
        compile_mode=cfg.compile_mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = CrossEntropyLoss()
    batch = to_device(synthetic_batch(cfg.micro_batch, cfg.depth, cfg.crop), device)
    n_params = sum(p.numel() for p in model.parameters())

    log(f"device={device} gpu={env['gpu']} amp={cfg.amp} "
        f"cuequivariance={cfg.cuequivariance} "
        f"compile={cfg.compile_mode} "
        f"checkpoint_triangles={cfg.checkpoint_triangles}")
    log(f"params={n_params/1e6:.1f}M shape=[{cfg.micro_batch}, {cfg.depth}, "
        f"{cfg.crop}] accum={cfg.accum} effective_batch={cfg.effective_batch}")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def optimizer_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.accum):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_on):
                loss = micro_step(
                    model, batch, criterion, cfg.accum,
                    checkpoint_triangles=cfg.checkpoint_triangles,
                )
        optimizer.step()
        return loss

    # Warmup is timed even though it is discarded. Under `torch.compile` the
    # first step pays for the compilation, and that cost is a fact about the
    # variant rather than noise to be thrown away: against a 10.5-day run a
    # ten-minute compile is free, and saying so plainly is what makes the
    # speedup quotable instead of caveated.
    warmup_times: list[float] = []
    for _ in range(cfg.warmup):
        sync()
        t0 = time.perf_counter()
        optimizer_step()
        sync()
        warmup_times.append(time.perf_counter() - t0)
    if warmup_times:
        log(f"warmup {sum(warmup_times):.1f}s "
            f"(first step {warmup_times[0]:.1f}s)")

    compiled = compile_stats() if cfg.compile_mode != "off" else None
    if compiled is not None:
        # Same refusal as the cuEquivariance check above, for the same reason.
        # Dynamo falls back to eager on errors it decides are recoverable, and
        # a run that compiled nothing would otherwise report the eager path
        # under the compiled label -- two identical numbers presented as a
        # finding.
        # `process_graphs`, not `unique_graphs`: from the sweep's second depth
        # onwards this model reuses code Dynamo compiled for the first one, so
        # zero *new* graphs is the normal case rather than a failure.
        if compiled["process_graphs"] == 0:
            raise RuntimeError(
                f"compile_mode={cfg.compile_mode!r} but Dynamo has captured no "
                "graphs anywhere in this process, so this would measure the "
                "eager path under the compiled label. Check for suppressed "
                "Dynamo errors."
            )
        log(f"compiled {compiled['unique_graphs']} new graphs "
            f"({compiled['process_graphs']} in this process), "
            f"{compiled['graph_breaks']} breaks")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_times: list[float] = []
    losses: list[float] = []
    for i in range(cfg.steps):
        sync()
        t0 = time.perf_counter()
        loss = optimizer_step()
        sync()  # the step is not over until the device says so
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        log(f"  step {i+1}/{cfg.steps}  {step_times[-1]:.3f}s  loss={losses[-1]:.4f}")

    median = sorted(step_times)[len(step_times) // 2]
    per_example = median / cfg.accum
    tokens_per_s = cfg.tokens_per_micro_batch / per_example
    phase = cfg.paper_phase
    # Per alignment, not per optimizer step: the two phases use effective
    # batches of 12 and 32, so their steps are not the same amount of work and
    # cannot be added together. Forward+backward passes can.
    phase_days = per_example * phase.examples / 86_400
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
    )

    result: dict[str, Any] = {
        "schema": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git": git if git is not None else git_info(),
        "environment": env,
        "precision": precision,
        "config": cfg.to_dict(),
        "model": {"params": n_params},
        "measurements": {
            "step_times_s": step_times,
            "median_step_s": median,
            "per_example_s": per_example,
            "tokens_per_s": tokens_per_s,
            "peak_gpu_gb": peak_gb,
            "losses": losses,
            "warmup_times_s": warmup_times,
        },
        "compile": compiled,
        "projection": {
            "phase": phase.key,
            "phase_steps": phase.steps,
            "phase_examples": phase.examples,
            "projected_phase_days": phase_days,
            "paper_days": REPORTED_DAYS,
            "paper_total_examples": TOTAL_EXAMPLES,
            "paper_s_per_example": PAPER_SECONDS_PER_EXAMPLE,
            "speedup_vs_paper": PAPER_SECONDS_PER_EXAMPLE / per_example,
        },
    }

    log(f"\nmedian optimizer step   {median:.3f} s")
    log(f"per alignment           {per_example*1000:.1f} ms "
        f"({tokens_per_s/1e3:.1f}k tokens/s)")
    if peak_gb is not None:
        log(f"peak GPU memory         {peak_gb:.1f} GB")
    log(f"{phase.key}: {phase.examples:,} alignments "
        f"({phase.steps:,} steps x {phase.effective_batch})  "
        f"{phase_days:.2f} days")
    log(f"paper: {REPORTED_DAYS} days for both phases, "
        f"{TOTAL_EXAMPLES:,} alignments, "
        f"{PAPER_SECONDS_PER_EXAMPLE*1000:.0f} ms each "
        f"(ratio {PAPER_SECONDS_PER_EXAMPLE / per_example:.2f}x)")
    return result


def build_parser() -> argparse.ArgumentParser:
    defaults = BenchConfig(device="cpu")  # device overridden below; rest are defaults
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--depth", type=int, default=defaults.depth, help="MSA depth (S)")
    ap.add_argument("--crop", type=int, default=defaults.crop, help="residue crop (N)")
    ap.add_argument("--micro-batch", type=int, default=defaults.micro_batch)
    ap.add_argument("--accum", type=int, default=defaults.accum,
                    help="gradient accumulation; effective batch = micro-batch * accum")
    ap.add_argument("--steps", type=int, default=defaults.steps,
                    help="timed optimizer steps")
    ap.add_argument("--warmup", type=int, default=defaults.warmup,
                    help="untimed optimizer steps first")
    ap.add_argument("--amp", choices=["bf16", "fp32"], default=defaults.amp)
    ap.add_argument("--lr", type=float, default=defaults.lr,
                    help="does not affect throughput")
    ap.add_argument("--float32-precision", choices=["highest", "high", "medium"],
                    default=defaults.float32_precision)
    ap.add_argument("--no-cuequivariance", dest="cuequivariance", action="store_false",
                    help="force the vanilla PyTorch triangle path")
    ap.add_argument("--compile", dest="compile_mode", choices=list(get_args(CompileMode)),
                    default=defaults.compile_mode,
                    help="torch.compile the repeated core-module leaves")
    ap.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    return ap


def main() -> None:
    args = vars(build_parser().parse_args())
    json_path: str | None = args.pop("json")
    # CPU hosts have no cuEquivariance; asking for it there is a mistake, not a
    # request, so the default follows the host rather than the dataclass.
    args["cuequivariance"] = args["cuequivariance"] and CUEQUIVARIANCE_PRESENT
    result = run(BenchConfig(**args))
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
