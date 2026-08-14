# ADR-0001: Eval module ports, records and package layout

## Status

Accepted

## Context

The eval module must reproduce three benchmarks (Figure 2 hetero-oligomer contact, CASP15 long-range contact, ProteinGym zero-shot DMS) and later accept variant models during training. The existing `evaluate/` code is untyped, mixes scoring with plotting, and uses pickled dicts for data. The map's standing decisions fix two ports and a numpy-only scoring core; this ADR pins the remaining interface choices.

Three coordinate frames complicate the design: model frame (alignment columns), structure frame (deposited residues), and scored frame (positions both resolve). The mapping between model and structure frame differs per benchmark — Figure 2 uses a boolean mask pair, CASP15 uses an index list, ProteinGym uses a scalar offset.

## Decisions

### D1: Two records unified by Projection

`ContactTarget` (Figure 2 and CASP15) and `DMSTarget` (ProteinGym). Each carries a `Projection` that maps model-frame positions to structure-frame positions. `Projection` is a base class exposing three members — `project_pred(NDArray) -> NDArray`, `project_truth(NDArray) -> NDArray`, and a `chain_break` property — with three concrete frozen-dataclass subtypes:

- `MaskPairProjection` (Figure 2): two boolean masks of different lengths (`msa_mask` in model frame, `cif_mask` in structure frame), optional `chain_break_model` position.
- `IndexProjection` (CASP15): integer index list into model-frame columns; truth side is identity (no `cif_mask` field — its absence is the declaration).
- `OffsetProjection` (ProteinGym): scalar offset between model-frame and structure-frame numbering.

The chain break is stored in the model frame on `MaskPairProjection`. `Projection.chain_break` returns the break in the scored frame (defaults to `None` for monomers). Callers never convert manually.

### D2: UnknownPolicy on the record

Ground-truth `-1` means "no structure." The paper's notebooks zero these before scoring (counted as true negatives, counted toward L). This is `AS_NEGATIVE` — reproduces the paper. `EXCLUDE` removes them from both numerator and denominator — the defensible metric, but does not reproduce the published 0.660.

`UnknownPolicy` is an enum field on `ContactTarget`, not a scoring-function parameter. Each benchmark manifest sets it once; the scoring core reads it.

### D3: No calibration port

Dropped. No published benchmark number depends on calibrated probabilities — P@K/P@L use rankings only, ProteinGym uses Spearman. The violin-plot probability analysis in Figure 2 is a supplementary visualization, not a scored metric. If calibration analysis is needed later, a `Calibrated` Protocol can be added in one session without changing the existing ports.

### D4: One adapter = one PairformerConfig

An adapter owns a `PairformerConfig` that specifies model identity, weights revision, device, and query-bias-ablation state. CASP15's qba-on / qba-off comparison becomes two named adapters, not one adapter with a variant-returning method. Predictions are returned in the model frame; the scoring core projects them via `Projection`.

### D5: Pydantic for config, dataclasses for records

Frozen `@dataclass` for data records (`ContactTarget`, `DMSTarget`, `Projection` subtypes) — they carry numpy arrays, and pydantic v2 would require `arbitrary_types_allowed` plus per-field validators for no gain. `pydantic_settings.BaseSettings` with `env_prefix="MSA_PAIRFORMER_EVAL_"` for runtime config (`EvalSettings`). No YAML, no untyped dicts. Pydantic is behind the `eval` optional extra; records use only stdlib + numpy.

### D6: Package layout

```
msa_pairformer/evaluate/          # namespace package (no __init__.py) — unchanged
    records.py                    # ContactTarget, DMSTarget, Projection, UnknownPolicy
    config.py                     # SamplingConfig, PairformerConfig, EvalSettings
    ports.py                      # ContactPredictor, VariantScorer
    core/                         # subpackage (has __init__.py, exports public API)
        __init__.py
        precision.py              # Region, select_pairs, p_at_k, p_at_l
    contacts.py                   # existing — untouched until metrics migrate
    plots.py                      # existing — untouched
    coevolution.py                # existing — untouched
    confind.py                    # existing — untouched
    proteingym.py                 # existing — untouched (vendored, 620 lines)
    structure.py                  # existing — untouched
```

`evaluate/` stays a namespace package (no `__init__.py`). `evaluate/core/` is a regular subpackage with `__init__.py` because it has a defined public API (the scoring functions). `core/` is numpy-only — no torch, no model, no I/O.

`compute_precision` (`contacts.py:18`) is not used by any benchmark and structurally cannot express the interface metric. It stays in the existing file untouched; it does not migrate into `core/`.

## Consequences

- New `eval` extra in `pyproject.toml`: `pydantic-settings>=2.3` (for `config.py` only; `records.py` uses stdlib + numpy).
- New modules must be classified in `tests/test_imports.py`'s `NOT_TIER_0` table.
- The scoring core (`core/precision.py`) is pure numpy and testable on hand-built matrices with no GPU.
- Metric implementation bodies belong to tickets #11 (Figure 2), #12 (CASP15), #13 (ProteinGym) — not this ADR.
