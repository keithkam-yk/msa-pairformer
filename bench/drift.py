"""Measure how far each execution path drifts from the recorded goldens.

The goldens in `tests/fixtures/golden.pt` were recorded from the upstream
implementation on CPU with the vanilla triangle path. Three configurations get
compared against them:

    cpu/vanilla    ~0        the existing correctness suite
    cuda/vanilla   isolates  float drift from moving to the GPU
    cuda/cuex      total     GPU drift *plus* the fused kernels

The middle row is what makes this interpretable. Without it a deviation under
cuEquivariance cannot be attributed -- it might be entirely CPU-to-GPU and have
nothing to do with the fused kernels. The cuEquivariance-specific tolerance is
row three against row two, and row two is itself the tolerance needed for
vanilla-on-GPU, which every future Modal run exercises.

**This module asserts nothing.** It exists to produce the numbers that
tolerances are then set from; a reporter that fails cannot report.

TF32 is pinned explicitly rather than inherited. Its default has moved across
torch versions, and with it on, GPU-vanilla drifts from CPU by roughly 1e-3
relative on its own -- enough to swamp the signal being measured. The setting
is recorded in the output because the resulting tolerances are only valid for
it.
"""

from __future__ import annotations

import torch

from bench.provenance import environment, git_info
from bench.step import CUEQUIVARIANCE_PRESENT, triangle_path

PACKAGE = "msa_pairformer"


def set_float32_precision(precision: str = "highest") -> dict[str, object]:
    """Pin fp32 matmul behaviour and report what was actually set."""
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = precision != "highest"
    torch.backends.cudnn.allow_tf32 = precision != "highest"
    return {
        "float32_matmul_precision": precision,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def deviation(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    """Max absolute and relative deviation, both as plain floats.

    Relative deviation is clamped rather than masked: near-zero reference
    values would otherwise dominate the maximum and say nothing useful.
    """
    got = got.detach().float().cpu()
    want = want.detach().float().cpu()
    diff = (got - want).abs()
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / want.abs().clamp_min(1e-6)).max().item(),
        "mean_abs": diff.mean().item(),
    }


def measure(
    device: torch.device,
    use_cuequivariance: bool,
    fixture: dict,
    shape_note: str = "fixture",
) -> dict[str, object]:
    """Replay every recorded case under one configuration.

    Records per-case deviations and any failure, rather than raising: one case
    that cannot run on a given path must not hide the numbers for the rest.
    """
    from tests.generate_fixtures import build_cases, param_checksum

    with triangle_path(use_cuequivariance) as effective:
        cases = dict(build_cases(PACKAGE, device=device, use_cuequivariance=effective))
        recorded = fixture["cases"]
        per_case: dict[str, object] = {}

        for name, expected in recorded.items():
            if name not in cases:
                per_case[name] = {"error": "no generator for this case"}
                continue
            try:
                module, _inputs, actual = cases[name]()
            except Exception as exc:
                per_case[name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue

            entry: dict[str, object] = {
                # Parameters are compared on the host in fp32, so this should
                # hold on any device. If it does not, the outputs below are
                # comparing different models and mean nothing.
                "checksum_matches": param_checksum(module) == expected["param_checksum"],
                "outputs": {},
            }
            for key, want in expected["outputs"].items():
                have = actual.get(key)
                if want is None or have is None:
                    entry["outputs"][key] = {"both_none": want is None and have is None}
                else:
                    entry["outputs"][key] = deviation(have, want.to(have.device))
            per_case[name] = entry

    return {
        "device": str(device),
        "requested_cuequivariance": use_cuequivariance,
        "effective_cuequivariance": effective,
        "shape": shape_note,
        "cases": per_case,
    }


def worst(config: dict) -> tuple[float, float]:
    """Largest absolute and relative deviation anywhere in one configuration."""
    abs_devs, rel_devs = [0.0], [0.0]
    for entry in config["cases"].values():
        if "outputs" not in entry:
            continue
        for dev in entry["outputs"].values():
            if "max_abs" in dev:
                abs_devs.append(dev["max_abs"])
                rel_devs.append(dev["max_rel"])
    return max(abs_devs), max(rel_devs)


def format_table(report: dict) -> str:
    lines = [f"{'configuration':<26}{'max abs':>12}{'max rel':>12}  notes"]
    # Deviations of the vanilla row for each device, so a cuex row can be
    # compared against its own baseline. `effective_cuequivariance` only says
    # the Python package imported (pairwise_operations.py:31); an ops extension
    # that fails to load against this torch is a later, separate failure. Two
    # bitwise-identical rows are the observable signature of that: the fused
    # kernels cannot reproduce the fallback exactly, so if they match, they did
    # not run. This is a note rather than an assertion -- a reporter that fails
    # cannot report.
    vanilla_by_device: dict[str, tuple[float, float]] = {}
    for config in report["configurations"]:
        label = f"{config['device']}/" + (
            "cuex" if config["effective_cuequivariance"] else "vanilla"
        )
        max_abs, max_rel = worst(config)
        notes = []
        if config["requested_cuequivariance"] and not config["effective_cuequivariance"]:
            notes.append("REQUESTED CUEX BUT RAN VANILLA")
        if config["effective_cuequivariance"]:
            baseline = vanilla_by_device.get(config["device"])
            if baseline == (max_abs, max_rel):
                notes.append("IDENTICAL TO VANILLA - FUSED KERNELS DID NOT ENGAGE")
        else:
            vanilla_by_device[config["device"]] = (max_abs, max_rel)
        bad_checksums = [
            n for n, e in config["cases"].items() if e.get("checksum_matches") is False
        ]
        if bad_checksums:
            notes.append(f"checksum mismatch: {len(bad_checksums)} case(s)")
        errors = [n for n, e in config["cases"].items() if "error" in e]
        if errors:
            notes.append(f"errors: {len(errors)}")
        lines.append(
            f"{label:<26}{max_abs:>12.3e}{max_rel:>12.3e}  {'; '.join(notes) or 'ok'}"
        )
    return "\n".join(lines)


def run(fixture_path: str = "tests/fixtures/golden.pt") -> dict:
    """Measure every available configuration and return the full report."""
    fixture = torch.load(fixture_path, weights_only=False)
    precision = set_float32_precision("highest")

    configurations = [measure(torch.device("cpu"), False, fixture)]
    if torch.cuda.is_available():
        configurations.append(measure(torch.device("cuda"), False, fixture))
        configurations.append(measure(torch.device("cuda"), True, fixture))

    return {
        "schema": 1,
        "git": git_info(),
        "environment": environment(),
        "precision": precision,
        "cuequivariance_present": CUEQUIVARIANCE_PRESENT,
        "fixture_metadata": fixture["metadata"],
        "configurations": configurations,
    }


if __name__ == "__main__":
    import json

    report = run()
    print(format_table(report))
    print()
    print(json.dumps(report["precision"], indent=2))
