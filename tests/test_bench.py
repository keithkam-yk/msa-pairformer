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

from bench.config import FINETUNE, PRETRAIN, TOTAL_EXAMPLES, BenchConfig
from bench.data import synthetic_batch
from bench.sweep import (
    DEFAULT_DEPTHS,
    MIN_POINTS_TO_QUOTE,
    curvature_bracket,
    example_seconds,
    fit_line,
    replace_depth,
    resolve_depths,
    whole_run_estimate,
)
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
    assert report["paper_estimate"]["depth"] == PRETRAIN.depth
    assert report["paper_estimate"]["is_measurement"] is False
    assert report["memory_ceiling"]["largest_depth_measured"] == 8


def test_fit_line_refuses_an_adjusted_r_squared_it_cannot_compute():
    """Three points against two parameters leaves one residual degree of
    freedom, so adjusted r^2 is undefined. Reporting None is the point: the
    first sweep quoted r^2 = 0.9945 off three points and meant almost nothing
    by it."""
    three = fit_line([64.0, 128.0, 192.0], [4.949, 5.643, 6.542])
    assert three["n_points"] == 3
    assert three["residual_dof"] == 1
    assert three["adjusted_r_squared"] is None
    assert three["r_squared"] > 0.99, "high r^2 on three points is the trap"

    five = fit_line([32.0, 64.0, 96.0, 128.0, 160.0], [2.0, 3.0, 4.0, 5.0, 6.0])
    assert five["adjusted_r_squared"] == pytest.approx(1.0)


def test_curvature_bracket_exposes_what_r_squared_hides():
    """The real sweep's three cuEquivariance points, whose r^2 was 0.9945. The
    parabola through them is 11% higher at depth 320, and that spread is the
    part of the estimate that is not evidence."""
    xs = [64.0, 128.0, 192.0]
    bracket = curvature_bracket(xs, [4.949, 5.643, 6.542], 320.0)

    assert bracket["linear"] == pytest.approx(8.10, abs=0.02)
    assert bracket["quadratic"] == pytest.approx(8.96, abs=0.02)
    # Convex: the middle point sits below the chord, so a straight line
    # extrapolates low. The first report's numbers were optimistic, not safe.
    assert bracket["quadratic"] > bracket["linear"]
    assert bracket["spread_fraction"] == pytest.approx(0.106, abs=0.01)


def test_curvature_bracket_is_tight_on_a_genuine_line():
    """The bracket has to be wide only when the data earns it, or it is just
    pessimism and nobody will read it."""
    xs = [32.0, 64.0, 96.0, 128.0, 160.0, 192.0]
    bracket = curvature_bracket(xs, [0.05 * x + 3.0 for x in xs], 320.0)

    assert bracket["linear"] == pytest.approx(19.0)
    assert bracket["spread_fraction"] < 1e-6


def test_sweep_refuses_to_quote_too_few_points():
    """A two-point sweep still produces an estimate -- it must not claim the
    estimate is usable."""
    report = sweep_run(BenchConfig(**TINY), depths=(4, 8), log=lambda *_: None)
    estimate = report["paper_estimate"]

    assert estimate["quotable"] is False
    assert estimate["estimated_step_s_low"] <= estimate["estimated_step_s_high"]
    assert estimate["extrapolation_reach"] == pytest.approx(PRETRAIN.depth / 8)


def test_resolve_depths_defaults_to_the_module_ladder():
    """The CLI default used to be a second copy of DEFAULT_DEPTHS. They
    diverged, and a rerun meant to replace a three-point fit measured the same
    three depths again."""
    assert resolve_depths("") == list(DEFAULT_DEPTHS)
    assert resolve_depths("   ") == list(DEFAULT_DEPTHS)
    assert resolve_depths("32,64, 96") == [32, 64, 96]

    # Enough points to make the quoting gate satisfiable at all, or the default
    # ladder cannot produce a usable estimate however well it fits.
    assert len(DEFAULT_DEPTHS) > MIN_POINTS_TO_QUOTE


def test_examples_ignore_how_the_batch_was_split():
    """The paper never states its micro-batch size, and does not need to: the
    work is steps x effective_batch either way. Counting optimizer steps
    instead would add 50,000 batches of 12 to 18,000 batches of 32 as though
    they were the same thing."""
    assert PRETRAIN.examples == 50_000 * 12
    assert FINETUNE.examples == 18_000 * 32
    assert TOTAL_EXAMPLES == 1_176_000

    # 10.5 days over that count. Quoted widely, so pinned here.
    from bench.config import PAPER_SECONDS_PER_EXAMPLE
    assert PAPER_SECONDS_PER_EXAMPLE == pytest.approx(0.7714, abs=5e-5)


def test_example_seconds_rescales_the_two_tracks_separately():
    """The pair track is O(crop^3) and the MSA track is linear in crop, so a
    single scale factor on the total would be wrong for both."""
    fit = {"slope": 0.01, "intercept": 6.0}

    same = example_seconds(fit, depth=320, accum=12, crop_from=312, crop_to=312)
    assert same == pytest.approx((6.0 + 3.2) / 12)

    bigger = example_seconds(fit, depth=320, accum=12, crop_from=312, crop_to=320)
    ratio = 320 / 312
    assert bigger == pytest.approx((6.0 * ratio**3 + 3.2 * ratio) / 12)
    assert bigger > same


def test_whole_run_covers_both_phases():
    """The reported 10.5 days is the whole run, so an estimate of one phase is
    not comparable to it."""
    fit = fit_line([32.0, 64.0, 96.0], [6.4, 6.7, 7.0])
    cfg = BenchConfig(device="cpu", cuequivariance=False)
    whole = whole_run_estimate(fit, cfg)

    assert set(whole["phases"]) == {"pretrain", "finetune"}
    assert whole["phases"]["finetune"]["crop_rescaled"] is True
    assert whole["phases"]["pretrain"]["crop_rescaled"] is False
    assert whole["total_examples"] == TOTAL_EXAMPLES
    assert whole["estimated_days"] == pytest.approx(
        sum(p["days"] for p in whole["phases"].values())
    )
    assert whole["is_measurement"] is False


def test_compile_mode_rejects_cuda_graphs_by_name():
    """`reduce-overhead` replays a static input buffer. Every shape here
    accumulates gradients over one reused batch tensor, so it would produce a
    time without producing the gradients that time claims to have cost."""
    with pytest.raises(ValueError, match="gradient accumulation"):
        BenchConfig(device="cpu", compile_mode="reduce-overhead")
    with pytest.raises(ValueError, match="compile_mode"):
        BenchConfig(device="cpu", compile_mode="fastest")

    for mode in ("off", "default", "max-autotune"):
        assert BenchConfig(device="cpu", compile_mode=mode).compile_mode == mode


def test_compile_targets_never_nest():
    """A compile boundary inside a region Dynamo is already tracing splits a
    graph that did not need splitting. `PairwiseBlock` contains a
    `PreLayerNorm(Transition)`, and both classes are targets."""
    from bench.step import compile_targets
    from msa_pairformer.pairwise_operations import PairwiseBlock

    block = PairwiseBlock(dim_pairwise=16, tri_mult_dim_hidden=8,
                          use_triangle_updates=False, use_pair_updates=False)
    parent = torch.nn.Module()
    parent.block = block

    names = compile_targets(parent)
    assert names == ["block"], names
    # The nested ones exist -- they are skipped, not absent.
    assert any(isinstance(m, torch.nn.Module) for _, m in block.named_modules())


def test_compile_targets_cover_every_repeated_leaf():
    """22 layers x 4 units. If a refactor renames a class out of
    COMPILE_TARGETS, the compiled variant quietly compiles less of the model
    and reports the difference as a smaller speedup."""
    from bench.step import build_model, compile_targets

    model = build_model(torch.device("cpu"), use_cuequivariance=False)
    names = compile_targets(model)
    per_layer = [n for n in names if n.startswith("core_stack.layers.")]
    assert len(per_layer) == 22 * 4
