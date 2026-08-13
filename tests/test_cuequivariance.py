"""Check that the fused triangle kernels agree with the PyTorch fallback.

`TriangleMultiplication`'s docstring states that `_vanilla_forward` "must align
with implementation in cuequivariance_torch.triangle_multiplicative_update in
order to serve as a fallback". Nothing verified that. It matters more than it
looks: cuEquivariance is a hard dependency on linux and switches on whenever
CUDA is present, so the fused path is what every real deployment runs, while
every golden in `fixtures/golden.pt` -- and every test on a developer laptop --
exercises only the fallback.

Why this and not recorded cuEquivariance goldens
------------------------------------------------
A recorded fixture would be pinned to the GPU that produced it: architecture,
cuEquivariance version, CUDA version, and Triton's precision defaults
(`_cuex_forward` passes `precision=None`). One recorded on an H100 would very
likely fail on a B200 for reasons that are not regressions. Comparing the two
paths side by side on whatever GPU is present tests the actual claim and stays
true across hardware.

Requires a CUDA host with cuequivariance_torch installed, so it skips
everywhere else -- including every machine this was developed on.

    modal run bench/modal_app.py::check   # runs this on a rented GPU
"""

import pytest
import torch

from bench.step import CUEQUIVARIANCE_PRESENT, triangle_path
from msa_pairformer.features import prepare_msa_masks
from msa_pairformer.nn import pairwise_operations
from msa_pairformer.nn.model import MSAPairformer
from msa_pairformer.tokens import aa2tok_d

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and CUEQUIVARIANCE_PRESENT),
    reason="needs a CUDA host with cuequivariance_torch",
)

# N is the paper's crop, and the size is load-bearing rather than realism for
# its own sake. Measured on an H100 at bf16, comparing the fused path against
# the fallback:
#
#     n=24    bitwise identical    fused 0.282ms  vanilla 0.358ms
#     n=128   max abs 1.6e-2       fused 0.506ms  vanilla 0.524ms
#     n=312   max abs 1.6e-2       fused 0.891ms  vanilla 2.099ms
#
# cuEquivariance declines small shapes and falls back to its own PyTorch
# reference -- which `_vanilla_forward` was written to mirror op for op, so the
# two agree bitwise. This test used to run at n=24 and therefore passed by
# comparing the fallback with itself. Anything that claims to cover the fused
# kernels has to be large enough for them to engage.
B, N, DIM_PAIRWISE = 1, 312, 256
S = 6

# Provisional. The fused kernel fuses differently and may reduce in a different
# order or precision, so exact agreement is not expected -- but disagreement
# beyond this would mean the fallback is not a faithful fallback. Tighten from
# the deviations the first GPU run reports rather than guessing downward.
RTOL, ATOL = 1e-2, 1e-3


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


def _pair_inputs(device):
    gen = torch.Generator(device="cpu").manual_seed(1234)
    pair = torch.randn(B, N, N, DIM_PAIRWISE, generator=gen).to(device)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -2:] = False
    pairwise_mask = (mask[:, :, None] & mask[:, None, :]).to(device)
    return pair, pairwise_mask


def _build(factory, use_cuequivariance, device, seed=0):
    """Same seed either side, so the comparison is of kernels and not weights.

    The flag is handed to the factory as well as set on the module global, and
    that is not belt-and-braces. `TriangleMultiplication.__init__` declares
    `use_cuequivariance: bool = CUEQUIVARIANCE_AVAILABLE` -- a default argument,
    evaluated once when the `def` executes at import time. On a CUDA host it is
    therefore frozen True, and `triangle_path` cannot move it however much it
    rewrites the global. Constructing without the argument silently gets the
    fused path even when vanilla was asked for, which is exactly how this test
    was failing: the "fallback" module it compared against was fused too.

    Modules that read the global at call time -- `PairwiseBlock`, and
    `generate_fixtures._triangle`, which passes it explicitly -- are unaffected.
    """
    with triangle_path(use_cuequivariance) as effective:
        torch.manual_seed(seed)
        module = factory(effective).to(device).eval()
    return module, effective


def _report(name, got, want):
    """Assert agreement, and report the deviation either way.

    The non-identity check comes first and is the more important of the two.
    A fused kernel cannot reproduce the fallback bitwise, so if it does, it did
    not run -- cuEquivariance declined the input and used its own PyTorch
    reference, which `_vanilla_forward` mirrors op for op. That is precisely how
    this file passed while testing nothing at N=24. Agreement is only evidence
    when there were two implementations to agree.
    """
    assert not torch.equal(got, want), (
        f"{name}: the fused and fallback paths are bitwise identical, so the "
        "fused kernel did not run -- cuEquivariance declined this input and "
        "fell back. This test cannot pass honestly at this shape or dtype; "
        "raise N (see the table at the top of this file) rather than trusting "
        "the green."
    )

    abs_dev = (got - want).abs().max().item()
    rel_dev = ((got - want).abs() / want.abs().clamp_min(1e-6)).max().item()
    print(f"{name}: max abs {abs_dev:.3e}, max rel {rel_dev:.3e}")
    torch.testing.assert_close(
        got, want, rtol=RTOL, atol=ATOL,
        msg=lambda m: f"{name}: cuEquivariance disagrees with the fallback "
                      f"(max abs {abs_dev:.3e}, max rel {rel_dev:.3e})\n{m}",
    )
    return abs_dev


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_triangle_multiplication_matches_fallback(device, direction):
    pair, pairwise_mask = _pair_inputs(device)

    def factory(use_cuequivariance):
        return pairwise_operations.TriangleMultiplication(
            dim_pairwise=DIM_PAIRWISE, dim_hidden=DIM_PAIRWISE, direction=direction,
            use_cuequivariance=use_cuequivariance,
        )

    fused, effective = _build(factory, True, device)
    assert effective, "expected the fused path to be selected"
    vanilla, _ = _build(factory, False, device)
    assert not vanilla.use_cuequivariance

    with torch.no_grad():
        got = fused(pair, pairwise_mask=pairwise_mask)
        want = vanilla(pair, pairwise_mask=pairwise_mask)
    _report(f"TriangleMultiplication[{direction}]", got, want)


def test_pairwise_block_matches_fallback(device):
    """The block wires both directions plus the transition, so it catches a
    disagreement in how the two triangle updates compose."""
    pair, pairwise_mask = _pair_inputs(device)

    # PairwiseBlock reads CUEQUIVARIANCE_AVAILABLE at call time and passes it
    # down, so it needs no explicit flag -- the argument is accepted and ignored
    # to keep one factory signature.
    def factory(_use_cuequivariance):
        return pairwise_operations.PairwiseBlock(dim_pairwise=DIM_PAIRWISE)

    fused, _ = _build(factory, True, device)
    vanilla, _ = _build(factory, False, device)

    with torch.no_grad():
        got = fused(pairwise_repr=pair, pairwise_mask=pairwise_mask)
        want = vanilla(pairwise_repr=pair, pairwise_mask=pairwise_mask)
    _report("PairwiseBlock", got, want)


def test_full_model_matches_fallback(device):
    """End to end: 22 layers of accumulated divergence is the number that
    actually matters, since that is what a user of the shipped weights gets."""
    torch.manual_seed(0)
    tokens = torch.randint(0, 20, (B, S, N))
    onehot = torch.nn.functional.one_hot(
        tokens, num_classes=len(aa2tok_d)
    ).float().to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokens)
    kwargs = {
        "msa": onehot, "mask": mask.to(device), "msa_mask": msa_mask.to(device),
        "full_mask": full_mask.to(device), "pairwise_mask": pairwise_mask.to(device),
        "return_cb_contacts": False, "return_confind_contacts": False,
        "store_msa_repr_cpu": False, "store_pairwise_repr_cpu": False,
    }

    # Same as PairwiseBlock: the model builds its blocks from the live global.
    def factory(_use_cuequivariance):
        return MSAPairformer()

    fused, _ = _build(factory, True, device)
    vanilla, _ = _build(factory, False, device)

    with torch.no_grad():
        got = fused(**kwargs)["logits"]
        want = vanilla(**kwargs)["logits"]
    _report("MSAPairformer.logits", got, want)
