# ADR-0001: Eval module ports, records and package layout

## Status

Accepted

## Context

The eval module must reproduce three benchmarks (Figure 2 hetero-oligomer contact, CASP15 long-range contact, ProteinGym zero-shot DMS) and later accept variant models during training. The existing `evaluate/` code is untyped, mixes scoring with plotting, and uses pickled dicts for data. The map's standing decisions fix two ports and a numpy-only scoring core; this ADR pins the remaining interface choices.

Three coordinate frames complicate the design: model frame (alignment columns), structure frame (deposited residues), and scored frame (positions both resolve). The mapping between model and structure frame differs per benchmark — Figure 2 uses a boolean mask pair, CASP15 uses an index list, ProteinGym uses a scalar offset.

## Decisions

### D1: Two records unified by Projection

`ContactTarget` (Figure 2 and CASP15) and `DMSTarget` (ProteinGym). Each carries a `Projection` that maps model-frame positions to structure-frame positions. `Projection` hides the mask-pair vs index-list vs offset representation behind a common interface.

The chain break is stored in the model frame on the record. `Projection` exposes a `chain_break` property that returns the break in the scored frame. Callers never convert manually.

### D2: UnknownPolicy on the record

Ground-truth `-1` means "no structure." The paper's notebooks zero these before scoring (counted as true negatives, counted toward L). This is `AS_NEGATIVE` — reproduces the paper. `EXCLUDE` removes them from both numerator and denominator — the defensible metric, but does not reproduce the published 0.660.

`UnknownPolicy` is an enum field on `ContactTarget`, not a scoring-function parameter. Each benchmark manifest sets it once; the scoring core reads it.

### D3: Calibrated as a separate Protocol

Three options were considered:
- A method on `ContactPredictor` that raises `NotImplementedError` for uncalibrated models — rejected: violates LSP, forces callers to try/except.
- A third port `CalibratedPredictor` — rejected: overweight for one optional capability.
- **Chosen**: a separate runtime-checkable `Calibrated` Protocol. An adapter may satisfy both `ContactPredictor` and `Calibrated`. Callers use `isinstance` to check before calling calibration methods.

This is consistent with the map's standing decision: "calibration metrics are an optional capability, not part of the port."

### D4: One adapter = one PairformerConfig

An adapter owns a `PairformerConfig` that specifies model identity, weights revision, device, and query-bias-ablation state. CASP15's qba-on / qba-off comparison becomes two named adapters, not one adapter with a variant-returning method. Predictions are returned in the model frame; the scoring core projects them via `Projection`.

### D5: Pydantic for all configuration

`pydantic.BaseModel` for data records (`ContactTarget`, `DMSTarget`, `Projection`). `pydantic_settings.BaseSettings` with `env_prefix="MSA_PAIRFORMER_EVAL_"` for runtime config (`EvalSettings`). No YAML, no untyped dicts. Both are behind a new `eval` optional extra.

### D6: Package layout

```
msa_pairformer/evaluate/          # namespace package (no __init__.py) — unchanged
    records.py                    # ContactTarget, DMSTarget, Projection, UnknownPolicy
    config.py                     # SamplingConfig, PairformerConfig, EvalSettings
    ports.py                      # ContactPredictor, VariantScorer, Calibrated
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

- New `eval` extra in `pyproject.toml`: `pydantic>=2.7`, `pydantic-settings>=2.3`.
- New modules must be classified in `tests/test_imports.py`'s `NOT_TIER_0` table.
- The scoring core (`core/precision.py`) is pure numpy and testable on hand-built matrices with no GPU.
- Metric implementation bodies belong to tickets #11 (Figure 2), #12 (CASP15), #13 (ProteinGym) — not this ADR.
