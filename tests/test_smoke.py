"""Minimal CPU smoke test for MSA Pairformer.

Runs a full forward pass on a tiny random MSA with randomly initialised weights
and pins the resulting output sums. The point is not accuracy -- it is to catch
refactors that silently change the numerics of the core stack.

The reference values below were recorded on the vanilla (non-cuEquivariance)
triangle path in float32 on CPU. They are therefore only meaningful for that
path: they do not cover the cuEquivariance kernels, bfloat16 autocast, or the
pretrained weights.

    pytest tests/test_smoke.py
"""

import pytest
import torch

from msa_pairformer.features import prepare_msa_masks
from msa_pairformer.model import MSAPairformer
from msa_pairformer.tokens import aa2tok_d

# Output sums for seed 0, S=8, N=24, float32, CPU, vanilla triangle path.
REFERENCE_SUMS = {
    "logits": 26.072940,
    "predicted_cb_contacts": 325.531852,
    "predicted_confind_contacts": 173.531063,
}
TOLERANCE = 1e-4

NUM_SEQS, SEQ_LEN, SEED = 8, 24, 0


@pytest.fixture(scope="module")
def forward_outputs():
    """One forward pass shared by every assertion below.

    Module-scoped because each parametrised case would otherwise rebuild a
    111M-parameter model and re-run the whole stack for a single sum.
    """
    torch.manual_seed(SEED)
    tokens = torch.randint(0, 20, (1, NUM_SEQS, SEQ_LEN))
    onehot = torch.nn.functional.one_hot(tokens, num_classes=len(aa2tok_d)).float()
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokens)

    torch.manual_seed(SEED)
    model = MSAPairformer().eval()

    with torch.no_grad():
        return model(
            msa=onehot,
            mask=mask,
            msa_mask=msa_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            return_cb_contacts=True,
            return_confind_contacts=True,
        )


@pytest.mark.parametrize(("key", "expected"), sorted(REFERENCE_SUMS.items()))
def test_forward_matches_reference(forward_outputs, key, expected):
    actual = forward_outputs[key].double().sum().item()
    assert actual == pytest.approx(expected, abs=TOLERANCE)
