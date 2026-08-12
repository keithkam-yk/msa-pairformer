"""Run the throughput harness and the drift measurement on a rented H100.

A thin adapter around `bench.train.run` and `bench.drift.run` -- all logic lives
there, so the same code path is exercised locally on CPU and remotely on the
GPU. This file only defines the container, requests the GPU, and moves results
back.

    modal run bench/modal_app.py::check       # correctness + golden drift
    modal run bench/modal_app.py::check --gpu L4      # same, cheaper card
    modal run bench/modal_app.py::sweep       # THE baseline: depths + estimate
    modal run bench/modal_app.py::main        # one fixed shape, both variants
    modal run bench/modal_app.py::main --variants cuequivariance --steps 10

`sweep` is the baseline to quote. `main` measures a single shape, which is only
useful when that shape fits -- the paper's does not fit on an 80 GB H100, so a
bare `main` at default settings OOMs by design rather than by accident.

The entrypoint is never optional. Modal only infers one when a file defines a
single local entrypoint, and this file has two; the bare form fails with a
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
    PAPER_ACCUM,
    PAPER_CROP,
    PAPER_DAYS,
    PAPER_DEPTH,
    PAPER_SECONDS_PER_STEP,
    Amp,
    BenchConfig,
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
def sweep(
    # Empty resolves to bench.sweep.DEFAULT_DEPTHS rather than repeating it
    # here; the two copies diverged once already.
    depths: str = "",
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

    ladder = resolve_depths(depths)
    print(f"depths: {ladder}")
    requested = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = set(requested) - {"vanilla", "cuequivariance"}
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")

    reports: dict[str, Any] = {}
    for name in requested:
        print(f"\n########## {name} ##########")
        cfg = BenchConfig(
            device="cuda", crop=PAPER_CROP, micro_batch=1, accum=PAPER_ACCUM,
            steps=steps, warmup=warmup, amp="bf16",
            cuequivariance=(name == "cuequivariance"),
        )
        reports[name] = depth_sweep.with_options(gpu=gpu).remote(
            cfg.to_dict(), git, ladder
        )
        print(format_table(reports[name]))

    if {"vanilla", "cuequivariance"} <= reports.keys():
        _compare_variants(reports)

    _write(out or _results_path(gpu, "sweep"),
           {"gpu_requested": gpu, "git": git, "variants": reports})


def _compare_variants(reports: dict[str, Any]) -> None:
    """What the fused kernels buy, per depth and at the paper's shape."""
    fused = {p["depth"]: p for p in reports["cuequivariance"]["points"] if not p["oom"]}
    plain = {p["depth"]: p for p in reports["vanilla"]["points"] if not p["oom"]}
    shared = sorted(set(fused) & set(plain))
    if shared:
        print(f"\n{'depth':>7}{'vanilla s':>12}{'cuex s':>10}{'speedup':>10}")
        for depth in shared:
            v, c = plain[depth]["median_step_s"], fused[depth]["median_step_s"]
            print(f"{depth:>7}{v:>12.3f}{c:>10.3f}{v/c:>9.2f}x")

    estimates = {k: r["paper_estimate"] for k, r in reports.items()}
    if all(estimates.values()):
        print(f"\nestimated at depth {PAPER_DEPTH} (not measured):")
        for name, est in estimates.items():
            print(f"  {name:<16}{est['estimated_step_s']:>8.2f} s/step"
                  f"{est['estimated_days']:>8.2f} days"
                  f"{est['ratio_vs_paper']:>8.2f}x vs paper")


@app.local_entrypoint()
def main(
    depth: int = 320,
    crop: int = 312,
    micro_batch: int = 1,
    accum: int = 12,
    steps: int = 5,
    warmup: int = 2,
    amp: str = "bf16",
    variants: str = "vanilla,cuequivariance",
    verify_first: bool = True,
    checkpoint_triangles: bool = True,
    gpu: str = DEFAULT_GPU,
    out: str = "",
) -> None:
    from bench.provenance import git_info

    git = git_info()
    if git["dirty"]:
        print("WARNING: working tree is dirty; this result is not reproducible "
              "from the recorded commit alone.\n")

    requested = [v.strip() for v in variants.split(",") if v.strip()]

    if verify_first:
        print("verifying container ...")
        res = correctness.with_options(gpu=gpu).remote(with_drift=False)
        print(f"  {res['gpu']}  torch {res['torch']}  cuda {res['cuda']}")
        print("  " + res["output"].strip().splitlines()[-1])
        if res["returncode"] != 0:
            print("\ncorrectness suite FAILED in the container -- not benchmarking.")
            print(res["output"])
            raise SystemExit(1)
        if "cuequivariance" in requested and not res["cuequivariance_present"]:
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
    def config(cuequivariance: bool) -> BenchConfig:
        return BenchConfig(
            device="cuda", depth=depth, crop=crop, micro_batch=micro_batch,
            accum=accum, steps=steps, warmup=warmup,
            # Modal's CLI hands every flag over as a plain str; BenchConfig's
            # __post_init__ is what actually rejects a bad value.
            amp=cast("Amp", amp),
            cuequivariance=cuequivariance,
            checkpoint_triangles=checkpoint_triangles,
        )

    configs = {"vanilla": config(False), "cuequivariance": config(True)}
    unknown = set(requested) - set(configs)
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")

    results: dict[str, Any] = {}
    for name in requested:
        print(f"=== {name} ===")
        results[name] = benchmark.with_options(gpu=gpu).remote(
            configs[name].to_dict(), git
        )
        print()

    print(f"{'variant':<18}{'s/step':>9}{'tokens/s':>12}{'peak GB':>10}"
          f"{'proj. days':>12}{'vs paper':>10}")
    for name, res in results.items():
        m, p = res["measurements"], res["projection"]
        peak = f"{m['peak_gpu_gb']:.1f}" if m["peak_gpu_gb"] is not None else "-"
        print(f"{name:<18}{m['median_step_s']:>9.3f}{m['tokens_per_s']/1e3:>11.1f}k"
              f"{peak:>10}{p['projected_days']:>12.2f}{p['speedup_vs_paper']:>9.1f}x")
    print(f"{'paper (1xH100)':<18}{PAPER_SECONDS_PER_STEP:>9.1f}{'-':>12}{'-':>10}"
          f"{PAPER_DAYS:>12.1f}{1.0:>9.1f}x")

    if {"vanilla", "cuequivariance"} <= results.keys():
        fused = results["cuequivariance"]["measurements"]["median_step_s"]
        plain = results["vanilla"]["measurements"]["median_step_s"]
        print(f"\nfused triangle kernels are {plain/fused:.2f}x the vanilla path "
              f"({plain:.3f}s -> {fused:.3f}s per optimizer step)")

    _write(
        out or _results_path(gpu, "baseline"),
        {"gpu_requested": gpu, "git": git, "variants": results},
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
