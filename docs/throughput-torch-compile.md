# Training throughput: torch.compile

This document records what `torch.compile` gives on top of the baseline in
[throughput-baseline.md](throughput-baseline.md). It gives the design of the
compiled variant, the check that ran before the measurement, the measurements,
and the new estimates for the paper's two training phases.

The text uses ASD-STE100 Simplified Technical English. Sentences are short. Each
sentence gives one idea. The tenses are simple.

Read the baseline document first. This document does not repeat the paper's
numbers, the memory limit, or the method of the extrapolation.

## Technical names

This document adds these names to the names in the baseline document.

| Name | Definition |
| --- | --- |
| compiled variant | A measurement with `torch.compile` on |
| eager variant | A measurement with `torch.compile` off |
| Dynamo | The part of `torch.compile` that captures Python code as a graph |
| Inductor | The part of `torch.compile` that generates kernels from that graph |
| graph break | A point where Dynamo stops the graph and gives control back to Python |
| compiled leaf | One of the four modules that `CoreModule` repeats 22 times |
| compile probe | The small test that runs before a full depth sweep |

## 1. What we compiled, and why not the model

`torch.compile` is applied to the four modules that `CoreModule` repeats at each
of its 22 layers:

| Compiled leaf | Track |
| --- | --- |
| `MSAPairWeightedAveraging` | MSA |
| `PreLayerNorm(Transition)` | MSA |
| `OuterProduct` | MSA to pair |
| `PairwiseBlock` | pair |

We did not compile `MSAPairformer` itself. Its `forward` is a Python loop across
`self.layers`. The loop holds dictionary accumulation, `del` statements,
`enumerate`, membership tests against index lists, and a conditional `break`.
Dynamo unrolls all 22 layers into one graph and adds a guard for each of those
constructs. The compile time is then long, and an unrelated argument invalidates
the graph. The repeated leaves hold the arithmetic and hold none of that Python.
One compilation of a leaf serves all 22 layers.

`PreLayerNorm` is compiled in place of the `Transition` inside it. A LayerNorm
in front of a linear layer is the pair that Inductor fuses. A compile boundary
between them discards that fusion.

A compiled leaf inside another compiled leaf is skipped. `PairwiseBlock`
contains a `PreLayerNorm`. Two boundaries would split a graph that needs no
split.

We use `Module.compile()` and not `torch.compile(module)`. The first changes the
call path of the module and keeps the module in the tree. The second returns a
wrapper, and the wrapper adds `_orig_mod.` to every parameter name. A compiled
run must stay comparable with every other run.

`reduce-overhead` is not an available mode. That mode replays CUDA graphs across
a static input buffer. Every shape here accumulates gradients across one reused
batch tensor. The mode gives a time, but the gradients that the time claims are
not the gradients that the run computes.

## 2. Four variants, not two

| Variant | Triangle path | `torch.compile` |
| --- | --- | --- |
| `vanilla` | PyTorch | off |
| `cuequivariance` | fused kernels | off |
| `vanilla+compile` | PyTorch | on |
| `cuequivariance+compile` | fused kernels | on |

The compiled cuEquivariance variant is the one to use. The compiled PyTorch
variant is the control. Without the control, a gain in the first variant has two
possible causes and no way to separate them: Inductor makes the model faster, or
the fused kernels make the model faster.

## 3. The compile probe

A full depth sweep is four variants at seven MSA depths. Each point builds a
model of 111 million parameters. Two questions decide if that cost is
justified, and a shape of one alignment at depth 32 and crop 64 answers both.

Data: [bench/results/h100-compile-probe.json](../bench/results/h100-compile-probe.json).

| Variant | First pass (s) | Second pass (s) | Peak memory (GB) | New graphs | Graph breaks |
| --- | --- | --- | --- | --- | --- |
| `vanilla` | 1.89 | 0.20 | 3.29 | – | – |
| `vanilla+compile` | 24.23 | 0.08 | 2.45 | 4 | 0 |
| `cuequivariance` | 0.36 | 0.20 | 4.14 | – | – |
| `cuequivariance+compile` | 7.49 | 0.13 | 2.46 | 1 | 0 |

These times are not throughput. The shape is small and there is no optimizer
step. The numbers show that the code runs.

Three results:

- **Compilation survives the activation checkpointing.** `PairwiseBlock` puts
  both triangle updates inside `torch.utils.checkpoint`. That was the largest
  risk, because the checkpoints are not optional at these shapes.
- **Zero graph breaks on both triangle paths at this shape.** We expected breaks
  at each fused kernel. The compiled region would then be only the elementwise
  work between the kernels. Section 4 tests the same count at crop 312, because
  the probe alone cannot settle it: cuEquivariance declines small shapes and
  returns the fallback result without a warning, and crop 64 can be below that
  limit. A clean trace of the fallback says nothing about the fused kernel.
- **Peak memory decreases.** It decreases 0.84 GB on the PyTorch path and 1.68
  GB on the cuEquivariance path. The dangerous result was an increase: a
  compiled variant with a lower memory limit looks like "compilation causes an
  out-of-memory error", but is an interaction with the recompute policy.

The count of new graphs needs one note. All four variants run in one process.
Three of the four compiled leaves are identical across the two triangle paths.
The second compiled variant therefore uses code that Dynamo compiled for the
first one, and reports 1 new graph in place of 4. The process compiles 5 graphs
in total.

### Numerical agreement

Compilation changes the choice of kernel and the order of the arithmetic. The
golden fixtures in `tests/fixtures/golden.pt` cannot check the result. They are
recorded eagerly, one module at a time. The probe therefore compares each
compiled variant with its own eager variant, at equal weights and on one batch:

The probe ran twice. The two runs give the spread of a measurement that has one
pass in it:

| Comparison | Loss | Gradient norm |
| --- | --- | --- |
| `vanilla+compile` against `vanilla` | 2e-05 to 3e-04 | 3e-04 to 9e-04 |
| `cuequivariance+compile` against `cuequivariance` | 5e-05 to 1e-04 | 3e-05 to 3e-04 |

Both values are relative. bf16 autocast puts a floor of approximately 1e-4 under
any comparison of this type. All of these deviations are at that floor.
Agreement at this level is not identity, and a different kernel schedule cannot
give identity.

The spread across the two runs is a factor of 20, so no single value above is a
measurement. The range is the result: the deviation stays at the bf16 floor and
does not go above it. The memory decrease repeated exactly (0.84 GB and 1.68
GB), and the second-pass ratio nearly so (2.3 and 2.7 times, then 1.50 and 1.48
times).

## 4. The depth sweep

RESULTS_SWEEP

## 5. The new estimates

RESULTS_ESTIMATE

## 6. What this does not measure

RESULTS_LIMITS

## 7. How to repeat the measurement

The probe first. It costs approximately two minutes of GPU time:

```bash
uv run modal run bench/modal_app.py::probe
```

Then the four-variant sweep:

```bash
uv run modal run bench/modal_app.py::sweep --variants vanilla,cuequivariance,vanilla+compile,cuequivariance+compile
```

The local self-check needs no GPU:

```bash
uv run pytest
```

## 8. Data files

| File | Content |
| --- | --- |
| [bench/results/h100-compile-probe.json](../bench/results/h100-compile-probe.json) | The probe: graphs, graph breaks, memory and numerical agreement |
| [bench/results/h100-sweep-pretrain.json](../bench/results/h100-sweep-pretrain.json) | The four-variant depth sweep and the estimates |
