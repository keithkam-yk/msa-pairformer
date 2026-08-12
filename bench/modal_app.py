"""Run the throughput harness and the drift measurement on a rented H100.

A thin adapter around `bench.train.run` and `bench.drift.run` -- all logic lives
there, so the same code path is exercised locally on CPU and remotely on the
GPU. This file only defines the container, requests the GPU, and moves results
back.

    modal run bench/modal_app.py::check       # correctness + golden drift
    modal run bench/modal_app.py::check --gpu L4      # same, cheaper card
    modal run bench/modal_app.py::probe       # does torch.compile work here
    modal run bench/modal_app.py::sweep       # THE baseline: depths + estimate
    modal run bench/modal_app.py::main        # one fixed shape, chosen variants
    modal run bench/modal_app.py::sweep --variants cuequivariance,cuequivariance+compile

`sweep` is the baseline to quote. `main` measures a single shape, which is only
useful when that shape fits -- the paper's does not fit on an 80 GB H100, so a
bare `main` at default settings OOMs by design rather than by accident.

`probe` is the gate in front of any compiled sweep, and costs about two
minutes. A compiled ladder is four variants times seven depths of
111M-parameter builds; the probe answers at a trivial shape whether
compilation survives the checkpointed triangle updates and how much of the
model Dynamo captured, which is what decides whether the ladder is worth
renting.

The entrypoint is never optional. Modal only infers one when a file defines a
single local entrypoint, and this file has several; the bare form fails with a
listing rather than running `main`.

Run `check` before trusting anything else: it establishes both that the GPU
paths work and what deviation from the recorded goldens each one produces,
which is what the tolerances in `tests/test_correctness.py` are set from.

The GPU defaults to H100 because the point of the benchmark is comparability
against the paper's "10.5 days on a single H100"; measured on any other card
the number has no reference to sit against. Note Modal's H100 is SXM -- the
paper does not say which variant it used, and PCIe is meaningfully slower, so
that stays a caveat on any comparison.

`--gpu` exists mainly for `check`, where the question is whether the fused
kernels run at all rather than how fast anything is. Prefer Ampere or newer
(L4, A10G, L40S): the fused triangle path is Triton-generated and older cards
are not worth the debugging.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import modal

from bench.config import (
    PAPER_SECONDS_PER_EXAMPLE,
    PHASES,
    PRETRAIN,
    REPORTED_DAYS,
    Amp,
    BenchConfig,
    CompileMode,
)

REPO = Path(__file__).parent.parent

# The default, overridable per run with --gpu. H100 is the default because the
# point of the benchmark is comparability against the paper's "10.5 days on a
# single H100"; measured on another card the number has no reference to sit
# against.
#
# `check` is the exception and is why this is a flag at all: correctness needs
# *a* CUDA device, not a specific one. The fused kernels either run or they do
# not, and that answer is card-independent -- so a cheaper GPU verifies the
# path perfectly well. The drift *magnitudes* are not card-independent, which
# is why the device name is recorded in the report and in the output filename.
DEFAULT_GPU = "H100"

# The measurable variants, as (triangle path, compile mode).
#
# Four rather than two, because a single compiled number cannot be attributed.
# `cuequivariance+compile` is the one asked for and the one anyone running the
# shipped code would reach for, but the fused triangle kernels are opaque to
# Dynamo, so what compiles there is the elementwise work *between* them.
# `vanilla+compile` is the control that says whether Inductor can approach the
# hand-written kernels on the pair track itself. Without it, a win in the first
# is unattributable between the two effects.
VARIANTS: dict[str, tuple[bool, str]] = {
    "vanilla": (False, "off"),
    "cuequivariance": (True, "off"),
    "vanilla+compile": (False, "default"),
    "cuequivariance+compile": (True, "default"),
}
BASELINE_VARIANT = "vanilla"

# Dependencies of the modules the harness touches (model, dataset, and their
# imports), rather than the full project: no matplotlib, sklearn or tqdm.
# The cuequivariance stack is linux-only and so cannot be exercised locally on
# macOS -- on Modal it installs, and the fused triangle kernels engage. That
# makes this the first time that path runs at all, which is why `check` exists.
#
# torch is pinned exactly, and the constraint is numerical rather than hygienic:
# the goldens in tests/fixtures were recorded under this version, so a floating
# `torch>=2.5.0` would let the drift table conflate torch-version differences
# with the device and kernel differences it is trying to isolate. Every
# tolerance derived from it would then be unattributable.
#
# This list is a second source of truth alongside uv.lock, and the two can
# drift. If they do, "the goldens were recorded under 2.13.0" quietly becomes
# "local venv under one version, image under another" -- the exact failure this
# pin exists to prevent. TORCH must track the torch resolved in uv.lock:
#
#     uv lock && grep -A1 '^name = "torch"' uv.lock
TORCH = "2.13.0"
CUEQUIVARIANCE = "0.11.1"

# cu13, not the cu12 that pyproject.toml pins for linux. torch 2.13's PyPI wheel
# is built against CUDA 13 (it pulls nvidia-cudnn-cu13, nvidia-nccl-cu13), so
# the cu12 ops wheels resolve *alongside* a second, CUDA 12 stack -- both
# nvidia-cublas 13.1.1.3 and nvidia-cublas-cu12 12.9.2.10 in one image, with
# ops compiled against an ABI torch is not using. Model code only ever imports
# `cuequivariance_torch` (pairwise_operations.py:31), so which ops variant sits
# underneath is invisible to it, and matching torch's CUDA major costs nothing.
#
# uv_pip_install rather than pip_install: same resolver as the local uv.lock, so
# when the two lists are compared they were at least produced by the same
# machinery, and the layer builds in a fraction of the time.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"torch=={TORCH}",
        "numpy>=1.26,<2.0",
        "einops>=0.8.0",
        "einx>=0.3.0",
        "jaxtyping>=0.2.38",
        "environs>=14.5.0",
        "huggingface-hub>=0.24.0",
        "biopython>=1.85",
        "scipy>=1.8.0",
        "beartype>=0.19.0",
        "pytest>=8.0",
    )
    .uv_pip_install(
        f"cuequivariance=={CUEQUIVARIANCE}",
        f"cuequivariance-torch=={CUEQUIVARIANCE}",
        f"cuequivariance-ops-cu13=={CUEQUIVARIANCE}",
        f"cuequivariance-ops-torch-cu13=={CUEQUIVARIANCE}",
    )
    .add_local_python_source("msa_pairformer", "bench")
    .add_local_dir(REPO / "tests", remote_path="/root/tests")
)

app = modal.App("msa-pairformer-bench", image=image)


@app.function(gpu=DEFAULT_GPU, timeout=1800)
def correctness(with_drift: bool = True) -> dict[str, Any]:
    """Everything that answers "is the GPU path correct", in one container.

    The suite and the drift report were separate functions. They ask the same
    question at two strictnesses -- the suite asserts and tells you pass or
    fail, the drift table measures and tells you by how much -- and they share
    the expensive part, which is the cold start and building the model. Two
    trips to a GPU to answer one question was the wrong shape.

    Drift runs *first* and never raises, so when the suite fails the numbers
    that explain why are already in hand.
    """
    import os
    import subprocess

    import torch

    from bench.step import CUEQUIVARIANCE_PRESENT

    report = None
    if with_drift:
        from bench.drift import format_table
        from bench.drift import run as drift_run

        report = drift_run("/root/tests/fixtures/golden.pt")
        print(format_table(report))
        print()

    env = {**os.environ, "MSA_PAIRFORMER_TYPECHECK": "1"}
    proc = subprocess.run(
        # -s so the deviations test_cuequivariance prints on success reach the
        # log. pytest swallows stdout for passing tests, which is usually right
        # and is wrong here: those numbers are the only measurement of how far
        # the fused kernels sit from the fallback, and a passing run is exactly
        # when we want them.
        [sys.executable, "-m", "pytest", "/root/tests", "-q", "-rs", "-s"],
        capture_output=True, text=True, cwd="/root", env=env, check=False,
    )
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        # If the wheels failed to install this is False, every cuEquivariance
        # test skips, and the suite still reports green. Surfaced so callers can
        # refuse rather than silently trust a suite that checked nothing.
        "cuequivariance_present": CUEQUIVARIANCE_PRESENT,
        "returncode": proc.returncode,
        # Generous, because the tail is where the failures are and 4000 chars
        # truncated the list of failing tests out of a run that had 4 of them.
        "output": proc.stdout[-20000:] + proc.stderr[-4000:],
        "drift": report,
    }


@app.function(gpu=DEFAULT_GPU, timeout=7200)
def depth_sweep(
    cfg: dict[str, Any], git: dict[str, Any], depths: list[int]
) -> dict[str, Any]:
    """Every depth in one container, on purpose.

    One call per depth would pay a cold start and a fresh 111M-parameter build
    each time, and -- worse for the fit -- would spread the points across
    different physical GPUs. The extrapolation assumes the points differ only
    in depth, so they have to come off one card in one process.
    """
    from bench.sweep import format_table
    from bench.sweep import run as sweep_run

    report = sweep_run(BenchConfig.from_dict(cfg), git=git, depths=tuple(depths))
    print("\n" + format_table(report))
    return report


@app.function(gpu=DEFAULT_GPU, timeout=3600)
def compile_probe_fn(depth: int, crop: int, git: dict[str, Any]) -> dict[str, Any]:
    """All four variants in one container, at a shape small enough to be free.

    One container, for the same reason `depth_sweep` uses one: the compiled
    variants are compared against the eager ones, and a comparison across two
    physical cards measures the cards.
    """
    import torch

    from bench.probe import format_table
    from bench.probe import run as probe_run

    report = probe_run(torch.device("cuda"), depth=depth, crop=crop)
    report["git"] = git
    print("\n" + format_table(report))
    return report


@app.function(gpu=DEFAULT_GPU, timeout=3600)
def benchmark(cfg: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    """Measure one configuration.

    `cfg` crosses the wire as a dict and is revalidated by `BenchConfig` on
    arrival, so a malformed configuration fails in the container rather than
    producing a plausible-looking result. `git` is captured on the client --
    the image carries Python source but no `.git`.
    """
    from bench.train import run

    return run(BenchConfig.from_dict(cfg), git=git)


@app.local_entrypoint()
def check(
    with_drift: bool = True,
    gpu: str = DEFAULT_GPU,
    out: str = "",
) -> None:
    """Correctness on a GPU, without benchmarking anything.

    `tests/test_cuequivariance.py` cannot run anywhere else: the wheels are
    linux-only and the import is gated on `torch.cuda.is_available()`, so on a
    developer machine it always skips. Without this entrypoint its only
    execution path is the pre-flight inside `main` -- so it would run only when
    someone happened to benchmark, and not at all under --no-verify-first.

    Roughly a minute of GPU time. Run it after touching anything the fused
    kernels depend on. `--no-with-drift` skips the drift table when all you
    want is pass or fail.
    """
    res = correctness.with_options(gpu=gpu).remote(with_drift=with_drift)
    print(f"{res['gpu']}  torch {res['torch']}  cuda {res['cuda']}")
    print(f"cuequivariance present: {res['cuequivariance_present']}")
    print(res["output"])

    if res["drift"] is not None:
        _write(out or _results_path(gpu, "drift"), res["drift"])
        print("Set the tolerances in tests/test_correctness.py from the table "
              "above; they are valid only for this GPU, cuEquivariance version "
              "and precision setting, all recorded in the JSON.")

    if not res["cuequivariance_present"]:
        raise SystemExit(
            "\ncuEquivariance is absent, so every test of it SKIPPED and the "
            "suite is green without having checked anything. Fix the image."
        )
    if res["returncode"] != 0:
        raise SystemExit("\ntests failed in the container")


@app.local_entrypoint()
def probe(depth: int = 32, crop: int = 64, gpu: str = DEFAULT_GPU,
          out: str = "") -> None:
    """Whether the compiled variants work, and what compiled, in ~2 minutes.

    Run this before `sweep --variants ...+compile`. A full ladder is four
    variants times seven depths of 111M-parameter builds; this answers at a
    trivial shape whether compilation survives the checkpointed triangle
    updates, how much of the model Dynamo captured on each triangle path, and
    whether the compiled path computes the same loss and gradients.

    It reports timings, and they are not throughput. The shape is tiny and
    there is no optimizer step; the numbers say "this ran", not "this is
    faster".
    """
    from bench.probe import format_table
    from bench.provenance import git_info

    report = compile_probe_fn.with_options(gpu=gpu).remote(depth, crop, git_info())
    print("\n" + format_table(report))
    _write(out or _results_path(gpu, "compile-probe"), report)

    empty = [
        label for label, entry in report["variants"].items()
        if entry["compile"] is not None and entry["compile"]["process_graphs"] == 0
    ]
    if empty:
        raise SystemExit(
            f"\n{empty} requested compilation and Dynamo captured no graphs, so "
            "a sweep would measure the eager path under a compiled label."
        )


@app.local_entrypoint()
def sweep(
    # Empty resolves to bench.sweep.DEFAULT_DEPTHS rather than repeating it
    # here; the two copies diverged once already.
    depths: str = "",
    phase: str = PRETRAIN.key,
    variants: str = "vanilla,cuequivariance",
    steps: int = 5,
    warmup: int = 2,
    gpu: str = DEFAULT_GPU,
    out: str = "",
) -> None:
    """The comparable baseline, for a card the paper's shape does not fit on.

    Depth 320 needs ~94 GB (bench/results/h100-memory-depth.json) and this card
    has 79. The paper's own text puts their H100 at 96 GB, so their number
    stands; it just cannot be reproduced here directly. A bigger card would
    trade the memory problem for a bandwidth one -- an H200 moves ~4.8 TB/s
    against this card's ~3.35, and the workload is bandwidth-bound, so the
    result would flatter us for reasons unrelated to the code.

    Instead: measure the depths that fit, check the cost really is affine in
    depth, and extrapolate the one axis that has to move. Everything else is
    held at the paper's values, so depth is the only difference between what
    was measured and what it is compared against.
    """
    from bench.provenance import git_info
    from bench.sweep import format_table, resolve_depths

    git = git_info()
    if git["dirty"]:
        print("WARNING: working tree is dirty; this result is not reproducible "
              "from the recorded commit alone.\n")

    if phase not in PHASES:
        raise SystemExit(f"unknown phase {phase!r}, expected one of {sorted(PHASES)}")
    ladder = resolve_depths(depths)
    spec = PHASES[phase]
    print(f"phase: {spec.key}  depth cap {spec.depth}  crop {spec.crop}  "
          f"effective batch {spec.effective_batch}  "
          f"{spec.examples:,} alignments")
    print(f"depths: {ladder}")
    requested = _requested_variants(variants)

    reports: dict[str, Any] = {}
    for name in requested:
        print(f"\n########## {name} ##########")
        cuequivariance, compile_mode = VARIANTS[name]
        cfg = BenchConfig.for_phase(
            phase, device="cuda", micro_batch=1,
            steps=steps, warmup=warmup, amp="bf16",
            cuequivariance=cuequivariance,
            compile_mode=cast("CompileMode", compile_mode),
        )
        reports[name] = depth_sweep.with_options(gpu=gpu).remote(
            cfg.to_dict(), git, ladder
        )
        print(format_table(reports[name]))

    if len(reports) > 1:
        _compare_variants(reports)

    _write(out or _results_path(gpu, f"sweep-{phase}"),
           {"gpu_requested": gpu, "phase": phase, "git": git,
            "variants": reports})


def _requested_variants(variants: str) -> list[str]:
    requested = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = [v for v in requested if v not in VARIANTS]
    if unknown:
        raise SystemExit(
            f"unknown variants: {sorted(unknown)}, known: {sorted(VARIANTS)}"
        )
    return requested


def _points_by_depth(report: dict[str, Any]) -> dict[int, Any]:
    return {p["depth"]: p for p in report["points"] if not p["oom"]}


def _compare_variants(reports: dict[str, Any]) -> None:
    """What each variant buys over the baseline, per depth and over the run.

    Peak memory sits next to the times on purpose. Compilation changes which
    activations are kept, and the failure that would quietly ruin a sweep is a
    variant whose memory ceiling moved down -- that reads as "compile OOMs
    earlier" when it is really a recompute-policy interaction with the
    checkpointed triangle updates. A column is cheaper than finding out later.
    """
    base_name = BASELINE_VARIANT if BASELINE_VARIANT in reports else next(iter(reports))
    base = _points_by_depth(reports[base_name])
    others = [n for n in reports if n != base_name]

    for name in others:
        theirs = _points_by_depth(reports[name])
        shared = sorted(set(base) & set(theirs))
        if not shared:
            continue
        print(f"\n{name} vs {base_name}")
        print(f"{'depth':>7}{'base s':>10}{'this s':>10}{'speedup':>10}"
              f"{'base GB':>10}{'this GB':>10}")
        for depth in shared:
            b, t = base[depth], theirs[depth]
            bs, ts = b["median_step_s"], t["median_step_s"]
            print(f"{depth:>7}{bs:>10.3f}{ts:>10.3f}{bs/ts:>9.2f}x"
                  f"{_gb(b):>10}{_gb(t):>10}")

    runs = {k: r["whole_run"] for k, r in reports.items()}
    if all(runs.values()):
        print(f"\nwhole run, both phases (not measured). paper: "
              f"{REPORTED_DAYS} days, "
              f"{PAPER_SECONDS_PER_EXAMPLE*1000:.0f} ms per alignment")
        for name, run in runs.items():
            print(f"  {name:<24}{run['s_per_example']*1000:>6.0f} ms"
                  f"{run['estimated_days']:>8.2f} days"
                  f"{run['ratio_vs_paper']:>8.2f}x vs paper")


def _gb(point: dict[str, Any]) -> str:
    peak = point["peak_gpu_gb"]
    return f"{peak:.1f}" if peak is not None else "-"


@app.local_entrypoint()
def main(
    phase: str = "",
    depth: int = 0,
    crop: int = 0,
    micro_batch: int = 1,
    accum: int = 0,
    steps: int = 5,
    warmup: int = 2,
    amp: str = "bf16",
    variants: str = "vanilla,cuequivariance",
    verify_first: bool = True,
    checkpoint_triangles: bool = True,
    gpu: str = DEFAULT_GPU,
    out: str = "",
) -> None:
    """One shape, measured rather than extrapolated.

    `--phase pretrain` or `--phase finetune` takes depth, crop and effective
    batch from that phase together, and labels the projection with it. Passing
    them separately is how the old defaults came to hold depth 320 from
    fine-tuning beside crop 312 and batch 12 from pre-training.

    The compiled variants are why this entrypoint is now useful at the paper's
    shapes. Eagerly, pre-training needs about 83 GB and fine-tuning about 99
    GB, so neither fits on an 80 GB card and both had to be extrapolated from a
    depth sweep. Compilation moves those to roughly 56 GB and 67 GB
    (bench/results/h100-sweep-pretrain.json), so both phases can be measured
    directly.
    """
    from bench.provenance import git_info

    if phase:
        if phase not in PHASES:
            raise SystemExit(
                f"unknown phase {phase!r}, expected one of {sorted(PHASES)}"
            )
        spec = PHASES[phase]
        depth = depth or spec.depth
        crop = crop or spec.crop
        accum = accum or spec.effective_batch
    else:
        # The historical defaults, kept so an explicit call keeps working.
        phase = PRETRAIN.key
        depth = depth or 320
        crop = crop or 312
        accum = accum or 12

    git = git_info()
    if git["dirty"]:
        print("WARNING: working tree is dirty; this result is not reproducible "
              "from the recorded commit alone.\n")

    requested = _requested_variants(variants)

    if verify_first:
        print("verifying container ...")
        res = correctness.with_options(gpu=gpu).remote(with_drift=False)
        print(f"  {res['gpu']}  torch {res['torch']}  cuda {res['cuda']}")
        print("  " + res["output"].strip().splitlines()[-1])
        if res["returncode"] != 0:
            print("\ncorrectness suite FAILED in the container -- not benchmarking.")
            print(res["output"])
            raise SystemExit(1)
        wants_cuex = any(VARIANTS[name][0] for name in requested)
        if wants_cuex and not res["cuequivariance_present"]:
            raise SystemExit(
                "\ncuEquivariance is not present in the container, so the "
                "'cuequivariance' variant would silently measure the vanilla "
                "path and report two identical baselines as a finding. Fix the "
                "image, or run with --variants vanilla."
            )
        print()

    # Two baselines, not a baseline and an optimisation.
    #
    # `vanilla` is the closer proxy for the reported 10.5-day run: the released
    # checkpoint was trained in the old fused-projection parameterisation and
    # converted to the cuEquivariance layout afterwards (see
    # reformat_weights_triangle_updates.ipynb), so the training run almost
    # certainly did not use these kernels.
    #
    # `cuequivariance` is what anyone running the shipped code on a GPU gets
    # today -- it is a hard dependency on linux and switches on automatically.
    #
    # Reporting both separates "we are running better kernels than the training
    # run did" from "there is real headroom left", which a single number cannot.
    def config(name: str) -> BenchConfig:
        cuequivariance, compile_mode = VARIANTS[name]
        return BenchConfig(
            device="cuda", phase=phase,
            depth=depth, crop=crop, micro_batch=micro_batch,
            accum=accum, steps=steps, warmup=warmup,
            # Modal's CLI hands every flag over as a plain str; BenchConfig's
            # __post_init__ is what actually rejects a bad value.
            amp=cast("Amp", amp),
            cuequivariance=cuequivariance,
            compile_mode=cast("CompileMode", compile_mode),
            checkpoint_triangles=checkpoint_triangles,
        )

    results: dict[str, Any] = {}
    for name in requested:
        print(f"=== {name} ===")
        results[name] = benchmark.with_options(gpu=gpu).remote(
            config(name).to_dict(), git
        )
        print()

    print(f"{'variant':<24}{'s/step':>9}{'ms/align':>10}{'tokens/s':>12}"
          f"{'peak GB':>10}{'phase days':>12}{'vs paper':>10}")
    for name, res in results.items():
        m, p = res["measurements"], res["projection"]
        peak = f"{m['peak_gpu_gb']:.1f}" if m["peak_gpu_gb"] is not None else "-"
        print(f"{name:<24}{m['median_step_s']:>9.3f}"
              f"{m['per_example_s']*1000:>10.0f}{m['tokens_per_s']/1e3:>11.1f}k"
              f"{peak:>10}{p['projected_phase_days']:>12.2f}"
              f"{p['speedup_vs_paper']:>9.1f}x")
    print(f"{'paper (1xH100)':<24}{'-':>9}"
          f"{PAPER_SECONDS_PER_EXAMPLE*1000:>10.0f}{'-':>12}{'-':>10}"
          f"{REPORTED_DAYS:>12.1f}{1.0:>9.1f}x  (both phases)")

    if BASELINE_VARIANT in results and len(results) > 1:
        plain = results[BASELINE_VARIANT]["measurements"]["median_step_s"]
        for name, res in results.items():
            if name == BASELINE_VARIANT:
                continue
            theirs = res["measurements"]["median_step_s"]
            print(f"{name} is {plain/theirs:.2f}x {BASELINE_VARIANT} "
                  f"({plain:.3f}s -> {theirs:.3f}s per optimizer step)")

    _write(
        out or _results_path(gpu, f"measured-{phase}"),
        {"gpu_requested": gpu, "phase": phase, "shape": {
            "depth": depth, "crop": crop, "micro_batch": micro_batch,
            "accum": accum,
        }, "git": git, "variants": results},
    )


def _results_path(gpu: str, kind: str) -> str:
    """Name results after the card that produced them.

    Neither drift magnitudes nor throughput transfer between GPUs, so a fixed
    filename would let an L4 run silently overwrite an H100 one and leave two
    incomparable numbers looking like a before and after.
    """
    slug = gpu.lower().replace(":", "x").replace("/", "-")
    return f"bench/results/{slug}-{kind}.json"


def _write(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
