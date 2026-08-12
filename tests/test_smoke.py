"""Minimal CPU smoke test for MSA Pairformer.

Runs a full forward pass on a tiny random MSA with randomly initialised weights
and pins the resulting output sums. The point is not accuracy -- it is to catch
refactors that silently change the numerics of the core stack.

The reference values below were recorded on the vanilla (non-cuEquivariance)
triangle path in float32 on CPU. They are therefore only meaningful for that
path: they do not cover the cuEquivariance kernels, bfloat16 autocast, or the
pretrained weights.

Run directly:      python tests/test_smoke.py
Or under pytest:   pytest tests/test_smoke.py
"""

import torch

from msa_pairformer.dataset import aa2tok_d, prepare_msa_masks
from msa_pairformer.model import MSAPairformer

# Output sums for seed 0, S=8, N=24, float32, CPU, vanilla triangle path.
REFERENCE_SUMS = {
    "logits": 26.072940,
    "predicted_cb_contacts": 325.531852,
    "predicted_confind_contacts": 173.531063,
}
TOLERANCE = 1e-4


def build_inputs(num_seqs: int = 8, seq_len: int = 24, seed: int = 0):
    torch.manual_seed(seed)
    tokens = torch.randint(0, 20, (1, num_seqs, seq_len))
    onehot = torch.nn.functional.one_hot(tokens, num_classes=len(aa2tok_d)).float()
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokens)
    return onehot, mask, msa_mask, full_mask, pairwise_mask


def run_forward():
    onehot, mask, msa_mask, full_mask, pairwise_mask = build_inputs()

    torch.manual_seed(0)
    model = MSAPairformer()
    model.eval()

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


def test_forward_matches_reference():
    res = run_forward()
    for key, expected in REFERENCE_SUMS.items():
        actual = res[key].double().sum().item()
        assert abs(actual - expected) < TOLERANCE, (
            f"{key}: expected {expected:.6f}, got {actual:.6f} "
            f"(delta {actual - expected:.3e})"
        )


if __name__ == "__main__":
    res = run_forward()
    ok = True
    for key, expected in REFERENCE_SUMS.items():
        actual = res[key].double().sum().item()
        delta = actual - expected
        status = "ok" if abs(delta) < TOLERANCE else "MISMATCH"
        if status != "ok":
            ok = False
        print(f"{key:28s} shape={tuple(res[key].shape)} sum={actual:.6f} {status}")
    print("smoke test passed" if ok else "smoke test FAILED")
    raise SystemExit(0 if ok else 1)
