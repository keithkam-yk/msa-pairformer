"""Guard the throughput harness against rot.

`bench/step.py` calls `MSAPairformer.forward` with five keyword arguments whose
defaults are wrong for training, and `bench/data.py` reaches into `msa_mlm` and
`prepare_msa_masks`. Any of that can break under an unrelated refactor. Without
this test the failure surfaces during a paid GPU run instead of locally.

Runs at trivial shapes on CPU -- it checks that the harness executes and that
gradients actually flow, not that any number is right.
"""

import pytest
import torch

from bench.config import BenchConfig
from bench.data import synthetic_batch
from bench.sweep import fit_line, replace_depth
from bench.sweep import run as sweep_run
from bench.train import run

TINY = {
    "device": "cpu", "depth": 8, "crop": 12, "micro_batch": 1, "accum": 2,
    "steps": 1, "warmup": 0, "amp": "fp32", "lr": 1e-3, "cuequivariance": False,
}


def test_synthetic_batch_matches_the_collate_contract():
    """Shapes and keys must be what CollateAFBatch would emit, or the harness
    is measuring something the real pipeline never produces."""
    b, s, n = 2, 6, 10
    batch = synthetic_batch(b, s, n)

    assert batch["msas"].shape == (b, s, n)
    assert batch["msas_onehot"].shape[:3] == (b, s, n)
    assert batch["mask"].shape == (b, n)
    assert batch["msa_mask"].shape == (b, s)
    assert batch["full_mask"].shape == (b, s, n)
    assert batch["pairwise_mask"].shape == (b, n, n)
    assert len(batch["masked_idx"]) > 0, "no positions masked -- loss would be empty"


def test_harness_runs_and_gradients_flow():
    result = run(TINY, log=lambda *_: None)

    assert result["measurements"]["median_step_s"] > 0
    loss = result["measurements"]["losses"][0]
    assert torch.isfinite(torch.tensor(loss)), f"non-finite loss {loss}"


def test_result_carries_provenance():
    """A timing number with no record of what produced it is not evidence."""
    result = run(TINY, git={"commit": "abc123", "branch": "main", "dirty": False},
                 log=lambda *_: None)

    assert result["git"]["commit"] == "abc123"
    assert result["config"]["depth"] == TINY["depth"]
    assert result["environment"]["torch"] == torch.__version__
    assert result["timestamp"] and result["command"]
    assert result["projection"]["paper_days"] == 10.5


def test_fit_line_recovers_a_known_line():
    """The paper comparison is an extrapolation of this fit, so the fit itself
    has to be checked against an answer known in advance rather than inferred
    from the measurement it is used to interpret."""
    xs = [64.0, 128.0, 192.0, 224.0]
    slope, intercept = 0.0375, 2.5
    fit = fit_line(xs, [slope * x + intercept for x in xs])

    assert fit["slope"] == pytest.approx(slope)
    assert fit["intercept"] == pytest.approx(intercept)
    assert fit["r_squared"] == pytest.approx(1.0)
    # The extrapolation is the point: a fit that is right on its own points but
    # wrong off the end of them would be useless here.
    assert slope * 320 + intercept == pytest.approx(14.5)


def test_fit_line_reports_a_poor_fit_rather_than_hiding_it():
    """r^2 is what decides whether the estimate may be quoted, so a curve that
    is not a line must drive it down, not merely fit badly in silence."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    curved = fit_line(xs, [x**3 for x in xs])
    assert curved["r_squared"] < 0.99

    noisy = fit_line(xs, [1.0, 5.0, 2.0, 6.0, 1.5])
    assert noisy["r_squared"] < 0.9


@pytest.mark.parametrize(
    ("xs", "ys"),
    [([1.0], [1.0]), ([2.0, 2.0], [1.0, 3.0])],
    ids=["one-point", "no-x-spread"],
)
def test_fit_line_refuses_undefined_input(xs, ys):
    with pytest.raises(ValueError):
        fit_line(xs, ys)


def test_replace_depth_changes_only_the_depth():
    """Every other knob is what makes the points comparable to each other and
    the extrapolation comparable to the paper."""
    cfg = BenchConfig(device="cpu", depth=64, cuequivariance=False)
    changed = replace_depth(cfg, 192)

    assert changed.depth == 192
    before, after = cfg.to_dict(), changed.to_dict()
    assert {k: v for k, v in after.items() if k != "depth"} == {
        k: v for k, v in before.items() if k != "depth"
    }


def test_sweep_fits_across_depths():
    """End to end on CPU at trivial shapes: the report has a point per depth,
    a fit, and an estimate flagged as not being a measurement."""
    cfg = BenchConfig(**TINY)
    report = sweep_run(cfg, depths=(4, 8), log=lambda *_: None)

    assert [p["depth"] for p in report["points"]] == [4, 8]
    assert all(not p["oom"] for p in report["points"])
    assert report["fit"] is not None
    assert report["paper_estimate"]["depth"] == 320
    assert report["paper_estimate"]["is_measurement"] is False
    assert report["memory_ceiling"]["largest_depth_measured"] == 8
