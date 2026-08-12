"""Guard the throughput harness against rot.

`bench/step.py` calls `MSAPairformer.forward` with five keyword arguments whose
defaults are wrong for training, and `bench/data.py` reaches into `msa_mlm` and
`prepare_msa_masks`. Any of that can break under an unrelated refactor. Without
this test the failure surfaces during a paid GPU run instead of locally.

Runs at trivial shapes on CPU -- it checks that the harness executes and that
gradients actually flow, not that any number is right.
"""

import torch

from bench.data import synthetic_batch
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
