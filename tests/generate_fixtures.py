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

* `utils.compute_precision` -- returns None upstream (it never returns a value).
* `chunk_layer(..., low_mem=True)` -- upstream references an undefined
  `_chunk_slice`.
* The outer product's `chunk_size` path -- upstream's differential `_chunk`
  passes the wrong number of arguments to `_opm`.

Pinning those against upstream would pin bugs, so they are excluded; they need
ordinary unit tests instead.

Weights come from `torch.manual_seed(seed)` immediately before construction
rather than being stored, which keeps fixtures small. Each case records a
checksum over the module's parameters so that a change to *initialisation* fails
with a distinct, obvious message instead of masquerading as a numerical
regression.

Triangle multiplication is recorded on the **vanilla PyTorch path**. A CUDA host
takes `_cuex_forward` instead and will not reproduce these values bitwise.
"""

import argparse
import hashlib
import importlib
import platform
import sys

import torch

# Small shapes: enough to exercise every axis (batch, sequences, residues,
# channels) while keeping the fixture file tiny and the run fast.
B, S, N = 1, 6, 10
DIM_MSA, DIM_PAIRWISE = 464, 256


def param_checksum(module) -> str:
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd):
        h.update(key.encode())
        h.update(sd[key].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def make_inputs(seed: int = 1234):
    """Deterministic inputs shared by every case, including a padded tail so the
    masking paths are actually exercised rather than trivially all-ones."""
    gen = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -2:] = False  # pad the last two residues
    msa_mask = torch.ones(B, S, dtype=torch.bool)
    msa_mask[:, -1] = False  # pad the last sequence
    full_mask = msa_mask[:, :, None] & mask[:, None, :]
    pairwise_mask = mask[:, :, None] & mask[:, None, :]
    return {
        "msa": torch.randn(B, S, N, DIM_MSA, generator=gen),
        "pairwise_repr": torch.randn(B, N, N, DIM_PAIRWISE, generator=gen),
        "mask": mask,
        "msa_mask": msa_mask,
        "full_mask": full_mask,
        "pairwise_mask": pairwise_mask,
    }


def build_cases(pkg: str):
    """Yield (name, callable) where the callable returns (module, inputs, outputs).

    `pkg` is the package to import from, so the same script drives the upstream
    tree (MSA_Pairformer) and the current one (msa_pairformer).
    """
    core = importlib.import_module(f"{pkg}.core")
    pairwise_operations = importlib.import_module(f"{pkg}.pairwise_operations")
    outer_product = importlib.import_module(f"{pkg}.outer_product")
    positional_encoding = importlib.import_module(f"{pkg}.positional_encoding")
    regression = importlib.import_module(f"{pkg}.regression")
    model_mod = importlib.import_module(f"{pkg}.model")

    inp = make_inputs()

    def case_transition():
        torch.manual_seed(0)
        mod = core.Transition(dim=DIM_MSA).eval()
        args = {"x": inp["msa"]}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def case_swiglu():
        mod = core.SwiGLU().eval()
        gen = torch.Generator().manual_seed(7)
        args = {"x": torch.randn(B, S, N, 128, generator=gen)}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def case_pair_weighted_averaging():
        torch.manual_seed(0)
        mod = pairwise_operations.MSAPairWeightedAveraging(
            dim_msa=DIM_MSA, dim_pairwise=DIM_PAIRWISE,
            heads=8, dim_head=32, dropout=0.0, dropout_type="row",
        ).eval()
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
                direction=direction, use_cuequivariance=False,
            ).eval()
            args = {"pair_rep": inp["pairwise_repr"], "pairwise_mask": inp["pairwise_mask"]}
            with torch.no_grad():
                return mod, args, {"out": mod(**args)}
        return case

    def case_pairwise_block():
        torch.manual_seed(0)
        mod = pairwise_operations.PairwiseBlock(dim_pairwise=DIM_PAIRWISE).eval()
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
        ).eval()
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
        ).eval()
        # Scalar inputs, not tensors -- recorded as-is.
        args = {"batch_size": B, "seq_len": N, "complex_chain_break_indices": [[4]]}
        with torch.no_grad():
            out = mod(device=torch.device("cpu"), **args)
        return mod, args, {"out": out}

    def case_lm_head():
        torch.manual_seed(0)
        mod = regression.LMHead(DIM_MSA, 26).eval()
        args = {"msa_repr": inp["msa"]}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def case_contact_head():
        torch.manual_seed(0)
        mod = regression.LogisticRegressionContactHead(dim_pairwise=DIM_PAIRWISE).eval()
        args = {"pair_repr": inp["pairwise_repr"]}
        with torch.no_grad():
            return mod, args, {"out": mod(**args)}

    def _full_model_inputs():
        dataset = importlib.import_module(f"{pkg}.dataset")
        gen = torch.Generator().manual_seed(99)
        tokens = torch.randint(0, 20, (B, S, N), generator=gen)
        onehot = torch.nn.functional.one_hot(
            tokens, num_classes=len(dataset.aa2tok_d)
        ).float()
        mask, msa_mask, full_mask, pairwise_mask = dataset.prepare_msa_masks(tokens)
        return {
            "msa": onehot, "mask": mask, "msa_mask": msa_mask,
            "full_mask": full_mask, "pairwise_mask": pairwise_mask,
        }

    def case_model_forward():
        torch.manual_seed(0)
        mod = model_mod.MSAPairformer().eval()
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
        mod = model_mod.MSAPairformer().eval()
        args = _full_model_inputs()
        with torch.no_grad():
            res = mod.predict_cb_contacts(**args)
        return mod, args, {"predicted_cb_contacts": res["predicted_cb_contacts"]}

    def case_model_predict_confind():
        torch.manual_seed(0)
        mod = model_mod.MSAPairformer().eval()
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
            "param_checksum": param_checksum(module),
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
