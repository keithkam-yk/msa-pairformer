"""Verify that opt-in jaxtyping runtime validation actually validates.

These tests only mean anything when MSA_PAIRFORMER_TYPECHECK is set before the
package is imported, so each one is skipped otherwise. Run them with:

    MSA_PAIRFORMER_TYPECHECK=1 python tests/test_typecheck.py
    MSA_PAIRFORMER_TYPECHECK=1 pytest tests/test_typecheck.py
"""

import torch

from msa_pairformer.custom_typing import should_typecheck
from msa_pairformer.dataset import aa2tok_d, prepare_msa_masks
from msa_pairformer.model import MSAPairformer

try:
    from jaxtyping import TypeCheckError
except ImportError:  # pragma: no cover
    TypeCheckError = TypeError

SKIP_REASON = "set MSA_PAIRFORMER_TYPECHECK=1 to exercise runtime shape checking"


def _tiny_inputs(num_seqs: int = 4, seq_len: int = 12):
    torch.manual_seed(0)
    tokens = torch.randint(0, 20, (1, num_seqs, seq_len))
    onehot = torch.nn.functional.one_hot(tokens, num_classes=len(aa2tok_d)).float()
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokens)
    return onehot, mask, msa_mask, full_mask, pairwise_mask


def _model():
    torch.manual_seed(0)
    model = MSAPairformer()
    model.eval()
    return model


def _predict(model, msa, mask, msa_mask, full_mask, pairwise_mask):
    with torch.no_grad():
        return model.predict_cb_contacts(
            msa=msa,
            mask=mask,
            msa_mask=msa_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
        )


def test_valid_input_is_accepted():
    """A well-formed call must pass with checking on -- guards against a shim
    that rejects everything, which would make the negative tests meaningless."""
    if not should_typecheck:
        return SKIP_REASON
    res = _predict(_model(), *_tiny_inputs())
    assert res["predicted_cb_contacts"].shape == (1, 12, 12)
    return None


def test_wrong_rank_is_rejected():
    """MSA passed as [b, n, d] instead of [b, s, n, d]."""
    if not should_typecheck:
        return SKIP_REASON
    msa, mask, msa_mask, full_mask, pairwise_mask = _tiny_inputs()
    try:
        _predict(_model(), msa[:, 0], mask, msa_mask, full_mask, pairwise_mask)
    except TypeCheckError:
        return None
    raise AssertionError("expected TypeCheckError for a rank-3 MSA")


def test_inconsistent_axis_is_rejected():
    """Shapes are individually valid but 'n' disagrees between the MSA and the
    pair mask. This is the failure mode a plain isinstance check cannot catch."""
    if not should_typecheck:
        return SKIP_REASON
    msa, mask, msa_mask, full_mask, _ = _tiny_inputs()
    mismatched_pair_mask = torch.ones(1, 8, 8, dtype=torch.bool)  # n=8, not 12
    try:
        _predict(_model(), msa, mask, msa_mask, full_mask, mismatched_pair_mask)
    except TypeCheckError:
        return None
    raise AssertionError("expected TypeCheckError for inconsistent 'n' axis")


TESTS = [
    test_valid_input_is_accepted,
    test_wrong_rank_is_rejected,
    test_inconsistent_axis_is_rejected,
]

if __name__ == "__main__":
    print(f"MSA_PAIRFORMER_TYPECHECK active: {should_typecheck}")
    failures = 0
    for test in TESTS:
        try:
            skipped = test()
        except AssertionError as exc:
            print(f"  FAIL {test.__name__}: {exc}")
            failures += 1
        else:
            print(f"  {'SKIP' if skipped else 'ok  '} {test.__name__}")
    raise SystemExit(1 if failures else 0)
