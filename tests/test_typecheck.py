"""Verify that opt-in jaxtyping runtime validation actually validates.

These tests only mean anything when MSA_PAIRFORMER_TYPECHECK is set *before* the
package is imported -- `custom_typing` reads the environment at import time and
binds `typecheck` to `identity` when it is unset. They are skipped otherwise:

    MSA_PAIRFORMER_TYPECHECK=1 pytest tests/test_typecheck.py

`TypeCheckError` is imported unguarded on purpose. jaxtyping is a core
dependency, and the fallback this module used to carry (`except ImportError:
TypeCheckError = TypeError`) would have quietly widened the negative tests to
"raises some TypeError" -- which a plain wrong-argument bug also satisfies.
"""

import pytest
import torch
from jaxtyping import TypeCheckError

from msa_pairformer.custom_typing import should_typecheck
from msa_pairformer.dataset import aa2tok_d, prepare_msa_masks
from msa_pairformer.model import MSAPairformer

pytestmark = pytest.mark.skipif(
    not should_typecheck,
    reason="set MSA_PAIRFORMER_TYPECHECK=1 to exercise runtime shape checking",
)

NUM_SEQS, SEQ_LEN = 4, 12


@pytest.fixture(scope="module")
def inputs():
    """A well-formed (msa, mask, msa_mask, full_mask, pairwise_mask) tuple."""
    torch.manual_seed(0)
    tokens = torch.randint(0, 20, (1, NUM_SEQS, SEQ_LEN))
    onehot = torch.nn.functional.one_hot(tokens, num_classes=len(aa2tok_d)).float()
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokens)
    return onehot, mask, msa_mask, full_mask, pairwise_mask


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return MSAPairformer().eval()


def predict(model, msa, mask, msa_mask, full_mask, pairwise_mask):
    with torch.no_grad():
        return model.predict_cb_contacts(
            msa=msa,
            mask=mask,
            msa_mask=msa_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
        )


def test_valid_input_is_accepted(model, inputs):
    """A well-formed call must pass with checking on -- guards against a shim
    that rejects everything, which would make the negative tests meaningless."""
    res = predict(model, *inputs)
    assert res["predicted_cb_contacts"].shape == (1, SEQ_LEN, SEQ_LEN)


def test_wrong_rank_is_rejected(model, inputs):
    """MSA passed as [b, n, d] instead of [b, s, n, d]."""
    msa, mask, msa_mask, full_mask, pairwise_mask = inputs
    with pytest.raises(TypeCheckError):
        predict(model, msa[:, 0], mask, msa_mask, full_mask, pairwise_mask)


def test_inconsistent_axis_is_rejected(model, inputs):
    """Shapes are individually valid but 'n' disagrees between the MSA and the
    pair mask. This is the failure mode a plain isinstance check cannot catch."""
    msa, mask, msa_mask, full_mask, _ = inputs
    mismatched_pair_mask = torch.ones(1, SEQ_LEN - 4, SEQ_LEN - 4, dtype=torch.bool)
    with pytest.raises(TypeCheckError):
        predict(model, msa, mask, msa_mask, full_mask, mismatched_pair_mask)
