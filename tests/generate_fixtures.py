"""Generate golden-output fixtures for the correctness suite.

The fixtures are produced by running the **original upstream implementation**
(the tree as it stood before the rename/lint/typing work) and recording its
outputs. `tests/test_correctness.py` then replays the exact same inputs through
the current code and asserts the outputs match, so any refactor that changes the
model's numerics fails loudly.

Regenerating from upstream
--------------------------
Check the upstream commit out into a separate worktree and run this script
against it. Upstream cannot be imported on a CPU-only host without one fix --
`CUEQUIVARIANCE_AVAILABLE` is assigned only inside `if torch.cuda.is_available()`,
so importing raises NameError -- so that single line must be patched in the
reference tree first. Nothing else may be changed there::

    git worktree add --detach /tmp/upstream 8753635
    # prepend `CUEQUIVARIANCE_AVAILABLE = False` before the try: in
    # /tmp/upstream/MSA_Pairformer/pairwise_operations.py
    cd /tmp/upstream && PYTHONPATH=/tmp/upstream \\
        python <repo>/tests/generate_fixtures.py MSA_Pairformer -o <repo>/tests/fixtures/golden.pt

To regenerate from the current tree instead (accepting current behaviour as the
new baseline -- do this deliberately, not to make a red test go green)::

    python tests/generate_fixtures.py msa_pairformer

Scope
-----
Only the **forward-pass path** is covered. Three things are deliberately out of
scope because the upstream reference cannot execute them at all:

* `evaluate.contacts.compute_precision` -- returns None upstream (it never
  returns a value).
* `chunk_layer(..., low_mem=True)` -- upstream references an undefined
  `_chunk_slice`.
* The outer product's `chunk_size` path -- upstream's differential `_chunk`
  passes the wrong number of arguments to `_opm`.

Pinning those against upstream would pin bugs, so they are excluded; they need
ordinary unit tests instead.

Weights come from `torch.manual_seed(seed)` immediately before construction
rather than being stored, which keeps fixtures small. Each case records a
*fingerprint* of the module's parameters -- exact shapes and key names, plus
per-key statistics compared at a tolerance -- so that a change to
*initialisation* fails with a distinct, obvious message instead of
masquerading as a numerical regression. See `param_fingerprint` for why a
checksum could not do this job across architectures.

Triangle multiplication is recorded on the **vanilla PyTorch path**. A CUDA host
takes `_cuex_forward` instead and will not reproduce these values bitwise.
"""

import argparse
import importlib
import math
import platform
import sys

import torch

# Small shapes: enough to exercise every axis (batch, sequences, residues,
# channels) while keeping the fixture file tiny and the run fast.
B, S, N = 1, 6, 10
DIM_MSA, DIM_PAIRWISE = 464, 256


# How far two fingerprints of the same initialisation may sit apart. Sized for
# accumulated last-bit noise over a reduction, not for a real change: a
# different distribution, an extra layer or a reordered draw moves these
# statistics by orders of magnitude, never by 1e-5.
FINGERPRINT_RTOL = 1e-5
FINGERPRINT_ATOL = 1e-8


def param_fingerprint(module) -> dict[str, dict[str, object]]:
    """Summarise a module's parameters in a way that survives a rebuild.

    This replaced a SHA-256 over the raw bytes, which could not survive one:
    parameters that are *computed* rather than drawn -- `q_proj.weight +
    randn_like(...) * 0.1` in outer_product.py, and `normal_(std=0.1)` -- come
    out 1 ULP apart on x86 and arm, because the scaling vectorises differently.
    The values agree to seven significant figures; the hash of them does not
    agree at all, and a hash has no tolerance to spend. Four cases therefore
    failed in the linux container while passing on the macOS host that recorded
    them, which reads exactly like "someone changed the model's init".

    Shapes and key names still have to match exactly -- those are structural,
    and no floating-point argument excuses a changed one. The statistics are
    positive-definite or extremal on purpose: a plain sum can cancel to near
    zero, where a relative comparison stops meaning anything.
    """
    out: dict[str, dict[str, object]] = {}
    for key, tensor in module.state_dict().items():
        value = tensor.detach().float().cpu()
        entry: dict[str, object] = {"shape": tuple(value.shape)}
        if value.numel():
            entry.update({
                "abs_sum": value.abs().sum().item(),
                "sq_sum": value.square().sum().item(),
                "min": value.min().item(),
                "max": value.max().item(),
            })
        out[key] = entry
    return out


def compare_fingerprints(got: dict, want: dict) -> list[str]:
    """Return one message per disagreement; empty means the two match."""
    problems = []
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    if missing:
        problems.append(f"parameters missing: {missing}")
    if extra:
        problems.append(f"unexpected parameters: {extra}")

    for key in sorted(set(got) & set(want)):
        g, w = got[key], want[key]
        if tuple(g["shape"]) != tuple(w["shape"]):
            problems.append(f"{key}: shape {g['shape']} != {w['shape']}")
            continue
        for stat in ("abs_sum", "sq_sum", "min", "max"):
            if stat not in w:
                continue
            a, b = float(g[stat]), float(w[stat])
            if not math.isclose(a, b, rel_tol=FINGERPRINT_RTOL,
                                abs_tol=FINGERPRINT_ATOL):
                problems.append(f"{key}.{stat}: {a:.8g} != {b:.8g}")
    return problems


def make_inputs(seed: int = 1234, device: torch.device | None = None):
    """Deterministic inputs shared by every case, including a padded tail so the
    masking paths are actually exercised rather than trivially all-ones."""
    gen = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -2:] = False  # pad the last two residues
    msa_mask = torch.ones(B, S, dtype=torch.bool)
    msa_mask[:, -1] = False  # pad the last sequence
    full_mask = msa_mask[:, :, None] & mask[:, None, :]
    pairwise_mask = mask[:, :, None] & mask[:, None, :]
    device = device or torch.device("cpu")
    return {
        "msa": torch.randn(B, S, N, DIM_MSA, generator=gen).to(device),
        "pairwise_repr": torch.randn(B, N, N, DIM_PAIRWISE, generator=gen).to(device),
        "mask": mask.to(device),
        "msa_mask": msa_mask.to(device),
        "full_mask": full_mask.to(device),
        "pairwise_mask": pairwise_mask.to(device),
    }


def build_cases(
    pkg: str,
    device: torch.device | None = None,
    use_cuequivariance: bool | None = None,
):
    """Yield (name, callable) where the callable returns (module, inputs, outputs).

    `pkg` is the package to import from, so the same script drives the upstream
    tree (MSA_Pairformer) and the current one (msa_pairformer).

    `use_cuequivariance` defaults to whatever the ambient
    `CUEQUIVARIANCE_AVAILABLE` says, so `bench.step.triangle_path` controls the
    triangle cases the same way it controls every other module. The goldens
    themselves were recorded with it off; replaying them with it on is how the
    drift between the two implementations gets measured.
    """
    core = importlib.import_module(f"{pkg}.core")
    pairwise_operations = importlib.import_module(f"{pkg}.pairwise_operations")
    outer_product = importlib.import_module(f"{pkg}.outer_product")
    positional_encoding = importlib.import_module(f"{pkg}.positional_encoding")
    regression = importlib.import_module(f"{pkg}.regression")
    model_mod = importlib.import_module(f"{pkg}.model")

    device = device or torch.device("cpu")
    if use_cuequivariance is None:
        use_cuequivariance = pairwise_operations.CUEQUIVARIANCE_AVAILABLE
    inp = make_inputs(device=device)

    def case_transition():
        torch.manual_seed(0)
        mod = core.Transition(dim=DIM_MSA).to(device).eval()
        args = {"x": inp["msa"]}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def case_swiglu():
        mod = core.SwiGLU().to(device).eval()
        gen = torch.Generator().manual_seed(7)
        args = {"x": torch.randn(B, S, N, 128, generator=gen).to(device)}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def case_pair_weighted_averaging():
        torch.manual_seed(0)
        mod = pairwise_operations.MSAPairWeightedAveraging(
            dim_msa=DIM_MSA, dim_pairwise=DIM_PAIRWISE,
            heads=8, dim_head=32, dropout=0.0, dropout_type="row",
        ).to(device).eval()
        args = {
            "msa": inp["msa"], "pairwise_repr": inp["pairwise_repr"],
            "mask": inp["mask"], "pairwise_mask": inp["pairwise_mask"],
            "full_mask": inp["full_mask"],
        }
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def _triangle(direction):
        def case():
            torch.manual_seed(0)
            mod = pairwise_operations.TriangleMultiplication(
                dim_pairwise=DIM_PAIRWISE, dim_hidden=DIM_PAIRWISE,
                direction=direction, use_cuequivariance=use_cuequivariance,
            ).to(device).eval()
            args = {"pair_rep": inp["pairwise_repr"], "pairwise_mask": inp["pairwise_mask"]}
            with torch.no_grad():
                return mod, args, {"out": mod(**args)}
        return case

    def case_pairwise_block():
        torch.manual_seed(0)
        mod = pairwise_operations.PairwiseBlock(dim_pairwise=DIM_PAIRWISE).to(device).eval()
        args = {"pairwise_repr": inp["pairwise_repr"], "pairwise_mask": inp["pairwise_mask"]}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def case_outer_product():
        torch.manual_seed(0)
        mod = outer_product.OuterProduct(
            dim_msa=DIM_MSA, dim_pairwise=DIM_PAIRWISE, dim_opm_hidden=16,
            outer_product_flavor="presoftmax_differential_attention",
            seq_attn=True, dim_qk=128, chunk_size=None, return_seq_weights=True,
            lambda_init=torch.tensor(0.8, dtype=torch.float32), eps=1e-32,
        ).to(device).eval()
        args = {
            "msa": inp["msa"], "mask": inp["mask"], "msa_mask": inp["msa_mask"],
            "full_mask": inp["full_mask"], "pairwise_mask": inp["pairwise_mask"],
        }
        with torch.no_grad():
            pair_update, seq_weights = mod(**args)
        return mod, args, {"pair_update": pair_update, "seq_weights": seq_weights}

    def case_relative_position_encoding():
        torch.manual_seed(0)
        mod = positional_encoding.RelativePositionEncoding(
            dim_out=DIM_PAIRWISE, r_max=32, s_max=2,
        ).to(device).eval()
        # Scalar inputs, not tensors -- recorded as-is.
        args = {"batch_size": B, "seq_len": N, "complex_chain_break_indices": [[4]]}
        with torch.no_grad():
            out = mod(device=device, **args)
        return mod, args, {"out": out}

    def case_lm_head():
        torch.manual_seed(0)
        mod = regression.LMHead(DIM_MSA, 26).to(device).eval()
        args = {"msa_repr": inp["msa"]}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def case_contact_head():
        torch.manual_seed(0)
        mod = regression.LogisticRegressionContactHead(dim_pairwise=DIM_PAIRWISE).to(device).eval()
        args = {"pair_repr": inp["pairwise_repr"]}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def _full_model_inputs():
        # This tree split `dataset.py` into `tokens.py` and `features.py`; the
        # upstream tree still has the one module. Both paths are tried so the
        # same script keeps driving both, per the `pkg` argument's purpose.
        try:
            tokens_mod = importlib.import_module(f"{pkg}.tokens")
            features_mod = importlib.import_module(f"{pkg}.features")
        except ModuleNotFoundError:
            tokens_mod = features_mod = importlib.import_module(f"{pkg}.dataset")
        gen = torch.Generator().manual_seed(99)
        tokens = torch.randint(0, 20, (B, S, N), generator=gen)
        onehot = torch.nn.functional.one_hot(
            tokens, num_classes=len(tokens_mod.aa2tok_d)
        ).float()
        mask, msa_mask, full_mask, pairwise_mask = features_mod.prepare_msa_masks(tokens)
        return {
            "msa": onehot.to(device), "mask": mask.to(device),
            "msa_mask": msa_mask.to(device), "full_mask": full_mask.to(device),
            "pairwise_mask": pairwise_mask.to(device),
        }

    def case_model_forward():
        torch.manual_seed(0)
        mod = model_mod.MSAPairformer().to(device).eval()
        args = _full_model_inputs()
        with torch.no_grad():
            res = mod(**args, return_cb_contacts=True, return_confind_contacts=True)
        return mod, args, {
            "logits": res["logits"],
            "predicted_cb_contacts": res["predicted_cb_contacts"],
            "predicted_confind_contacts": res["predicted_confind_contacts"],
            "final_pairwise_repr": res["final_pairwise_repr"],
        }

    def case_model_predict_cb():
        torch.manual_seed(0)
        mod = model_mod.MSAPairformer().to(device).eval()
        args = _full_model_inputs()
        with torch.no_grad():
            res = mod.predict_cb_contacts(**args)
        return mod, args, {"predicted_cb_contacts": res["predicted_cb_contacts"]}

    def case_model_predict_confind():
        torch.manual_seed(0)
        mod = model_mod.MSAPairformer().to(device).eval()
        args = _full_model_inputs()
        with torch.no_grad():
            res = mod.predict_confind_contacts(**args)
        return mod, args, {"predicted_confind_contacts": res["predicted_confind_contacts"]}

    return [
        ("core.Transition", case_transition),
        ("core.SwiGLU", case_swiglu),
        ("pairwise_operations.MSAPairWeightedAveraging", case_pair_weighted_averaging),
        ("pairwise_operations.TriangleMultiplication[outgoing]", _triangle("outgoing")),
        ("pairwise_operations.TriangleMultiplication[incoming]", _triangle("incoming")),
        ("pairwise_operations.PairwiseBlock", case_pairwise_block),
        ("outer_product.OuterProduct[presoftmax_differential]", case_outer_product),
        ("positional_encoding.RelativePositionEncoding", case_relative_position_encoding),
        ("regression.LMHead", case_lm_head),
        ("regression.LogisticRegressionContactHead", case_contact_head),
        ("model.MSAPairformer.forward", case_model_forward),
        ("model.MSAPairformer.predict_cb_contacts", case_model_predict_cb),
        ("model.MSAPairformer.predict_confind_contacts", case_model_predict_confind),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("package", nargs="?", default="msa_pairformer",
                    help="package to import from (MSA_Pairformer for upstream)")
    ap.add_argument("-o", "--output", default="tests/fixtures/golden.pt")
    args = ap.parse_args()

    cases = {}
    for name, fn in build_cases(args.package):
        module, inputs, outputs = fn()
        cases[name] = {
            "inputs": inputs,
            "outputs": outputs,
            "param_fingerprint": param_fingerprint(module),
        }
        shapes = ", ".join(
            f"{k}{tuple(v.shape)}" for k, v in outputs.items() if torch.is_tensor(v)
        )
        print(f"  {name:56s} {shapes}")

    payload = {
        "metadata": {
            "source_package": args.package,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "triangle_path": "vanilla",
            "shapes": {"B": B, "S": S, "N": N,
                       "dim_msa": DIM_MSA, "dim_pairwise": DIM_PAIRWISE},
        },
        "cases": cases,
    }
    torch.save(payload, args.output)
    print(f"\nwrote {len(cases)} cases -> {args.output}")


if __name__ == "__main__":
    main()
