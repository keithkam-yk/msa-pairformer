# Training throughput: torch.compile

This document records what `torch.compile` gives on top of the baseline in
[throughput-baseline.md](throughput-baseline.md). It gives the design of the
compiled variant, the check that ran before the measurement, the measurements,
and the new estimates for the paper's two training phases.

The text uses ASD-STE100 Simplified Technical English. Sentences are short. Each
sentence gives one idea. The tenses are simple.

Read the baseline document first. This document does not repeat the paper's
numbers, the memory limit, or the method of the extrapolation.

Date of the measurements: 12 August 2026. Repository commit: `3b8d275`.

## Summary

`torch.compile` gives approximately two times the throughput and removes
approximately one third of the peak memory. The memory result is the more
useful one, because it changes what we can measure.

| Variant | Whole run | Compared with the paper |
| --- | --- | --- |
| `vanilla` (baseline document) | 831 ms → 11.32 days | 0.93 times |
| `cuequivariance` (baseline document) | 664 ms → 9.03 days | 1.16 times |
| `vanilla+compile` | 388 ms → 5.28 days | 1.99 times |
| `cuequivariance+compile` | **370 ms → 5.03 days** | **2.09 times** |
| The paper | 771 ms → 10.5 days | 1.00 times |

Four results:

1. The compiled variants use 56 GB at the pre-training shape and 66 GB at the
   fine-tuning shape. Eagerly the same shapes need 83 GB and 99 GB. **Both
   phases therefore fit on an 80 GB H100 for the first time**, and the two
   compiled rows above are measurements at those shapes. The two eager rows
   stay estimates.
2. The fused cuEquivariance kernels give 1.25 times without compilation and
   only 1.07 times with it. The two changes largely overlap, and they do not
   add.
3. The fused kernels go into the Dynamo graph. There are no graph breaks at the
   pre-training crop, on either triangle path.
4. The extrapolation method of the baseline document now has a check against a
   measurement. It agrees to 3 per cent on time at the same crop, and to 1.2
   per cent on memory.

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

The harness applies `torch.compile` to the four modules that `CoreModule`
repeats at each of its 22 layers:

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

The harness compiles `PreLayerNorm` in place of the `Transition` inside it. A LayerNorm
in front of a linear layer is the pair that Inductor fuses. A compile boundary
between them discards that fusion.

The harness skips a compiled leaf inside another compiled leaf. `PairwiseBlock`
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
compiled variant with its own eager variant, at equal weights and on one batch.

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

All four variants ran in one sweep, at one commit, on one physical H100 80GB
HBM3. Crop 312, gradient accumulation 12, micro-batch 1, bf16, activation
checkpointing on. The method is the method of the baseline document: 2 untimed
warmup steps, then 5 timed steps, then the median.

Data: [bench/results/h100-sweep-pretrain.json](../bench/results/h100-sweep-pretrain.json).

### Time for each optimizer step

| MSA depth | `vanilla` | `cuequivariance` | `vanilla+compile` | `cuequivariance+compile` |
| --- | --- | --- | --- | --- |
| 32 | 6.306 | 4.392 | 3.199 | 2.936 |
| 64 | 6.751 | 4.834 | 3.388 | 3.123 |
| 96 | 7.224 | 5.302 | 3.593 | 3.330 |
| 128 | 7.559 | 5.651 | 3.760 | 3.499 |
| 160 | 7.986 | 6.061 | 3.995 | 3.682 |
| 192 | 8.455 | 6.520 | 4.177 | 3.916 |
| 224 | out of memory | out of memory | 4.391 | 4.127 |

Values are seconds. The eager variants agree with the baseline document to
better than 1 per cent, at every depth. The two sweeps are different days and
different containers, so that agreement is the run-to-run stability of the
measurement.

**The graph breaks stay at zero at crop 312.** Both compiled variants report 0
breaks at every depth. Section 3 could not settle this, because cuEquivariance
declines small shapes and gives the fallback result silently. Crop 312 is the
pre-training crop and is large enough. The fused kernels therefore go into the
Dynamo graph, and the compiled region on that path is the whole
`PairwiseBlock`. It is not only the elementwise work between the kernels.

### Peak memory

| MSA depth | `vanilla` | `cuequivariance` | `vanilla+compile` | `cuequivariance+compile` |
| --- | --- | --- | --- | --- |
| 32 | 38.5 | 38.4 | 27.2 | 28.0 |
| 64 | 44.8 | 44.8 | 31.1 | 32.1 |
| 96 | 51.1 | 51.1 | 35.2 | 36.2 |
| 128 | 57.5 | 57.4 | 39.4 | 40.4 |
| 160 | 63.8 | 63.8 | 43.5 | 44.5 |
| 192 | 70.1 | 70.0 | 47.7 | 48.7 |
| 224 | out of memory | out of memory | 51.9 | 52.9 |

Values are GB. This is the larger result of the run, and section 5 gives the
consequence.

| Variant | Memory fit (GB) | At depth 256, crop 312 | At depth 320, crop 320 |
| --- | --- | --- | --- |
| `vanilla` | 0.1975 × depth + 32.17 | 82.7 | 99.5 |
| `cuequivariance` | 0.1975 × depth + 32.15 | 82.7 | 99.5 |
| `vanilla+compile` | 0.1288 × depth + 22.93 | 55.9 | 67.0 |
| `cuequivariance+compile` | 0.1298 × depth + 23.78 | 57.0 | 68.3 |

Compilation changes both terms. The intercept decreases 9 GB, and the slope
decreases 35 per cent. The slope is the MSA track. Inductor fuses the
elementwise chains there and does not write the intermediate tensors, so fewer
activations stay alive for the backward pass. The choice of triangle path does
not change the memory, either eagerly or compiled.

### The fits

| Variant | Points | Fit (s/step) | r² | Adjusted r² |
| --- | --- | --- | --- | --- |
| `vanilla` | 6 | 0.01320 × depth + 5.902 | 0.9982 | 0.9970 |
| `cuequivariance` | 6 | 0.01310 × depth + 3.993 | 0.9986 | 0.9977 |
| `vanilla+compile` | 7 | 0.00620 × depth + 2.992 | 0.9991 | 0.9986 |
| `cuequivariance+compile` | 7 | 0.00615 × depth + 2.729 | 0.9983 | 0.9974 |

Compilation approximately halves both terms of the time fit.

The compiled variants have seven points and the eager variants have six,
because depth 224 fits only with compilation. The two sets of fits therefore do
not have equal reach. The compiled fits reach 1.14 times beyond their largest
measured depth. The eager fits reach 1.33 times. That difference favours the
compiled variants.

To remove the advantage, we fit the compiled variants again across the same six
depths that the eager variants reached. The result moves less than 1 per cent:

| Variant | All points | Depths 32 to 192 only |
| --- | --- | --- |
| `vanilla+compile` | 410 ms, 5.57 days | 409 ms, 5.56 days |
| `cuequivariance+compile` | 386 ms, 5.25 days | 383 ms, 5.22 days |

The difference between the two ladders is thus too small to change any
conclusion. The ratios between variants in section 5 use the six shared depths.
The whole-run figures for the compiled variants do not use either fit: section 5
measures those two shapes directly. The estimates above remain useful as the
check on the method.

### The cost of the compilation

`bench` records the warmup time at each point. The compilation is in the first
warmup step. The excess above two measured steps is the cost:

| Variant | 32 | 64 | 96 | 128 | 160 | 192 | 224 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vanilla` | 3 | 1 | 1 | 1 | 0 | 1 | – | 6 |
| `vanilla+compile` | 29 | 20 | 10 | 0 | 0 | 0 | 0 | 59 |

Values are seconds. Dynamo compiles at the first depth and adds dynamic shapes
at the second and the third. After depth 128 it compiles nothing more. The
total for the whole ladder is 59 seconds. A training run has one shape, so it
pays this once. Against 10.5 days, 30 seconds does not appear in any total.

`cuequivariance+compile` shows 7 seconds, but that figure is not a cold start.
Modal gave it the same warm container as `vanilla+compile`, and three of the
four compiled leaves are identical across the triangle paths. It compiled only
`PairwiseBlock` again.

## 5. The new estimates

### The whole run

The unit is the alignment, for the reason the baseline document gives: the two
phases use effective batches of 12 and 32, so their optimizer steps are not the
same quantity. The paper processes 1,176,000 alignments in 10.5 days, or 771 ms
for each alignment.

| Variant | Pre-training | Fine-tuning | Whole run | Compared with the paper | Source |
| --- | --- | --- | --- | --- | --- |
| `vanilla` | 773 ms → 5.37 d | 892 ms → 5.94 d | **831 ms → 11.32 d** | 0.93 times | estimate |
| `cuequivariance` | 612 ms → 4.25 d | 717 ms → 4.78 d | **664 ms → 9.03 d** | 1.16 times | estimate |
| `vanilla+compile` | 370 ms → 2.57 d | 407 ms → 2.71 d | **388 ms → 5.28 d** | 1.99 times | measurement |
| `cuequivariance+compile` | 350 ms → 2.43 d | 390 ms → 2.60 d | **370 ms → 5.03 d** | 2.09 times | measurement |
| The paper | | | 771 ms → 10.5 d | 1.00 times | reported |

The two compiled rows are measurements at the exact shapes of the two phases.
The two eager rows stay estimates, because those shapes still do not fit on the
card. That difference is real, and the last column records it.

The best variant gives 2.09 times the throughput of the paper's run, at the
depth caps of both phases.

### What each change gives

The ratios below use the six depths that all four variants reached.

| Change | On the PyTorch path | On the cuEquivariance path |
| --- | --- | --- |
| `torch.compile` | 2.04 times | 1.73 times |

| Change | Eagerly | With `torch.compile` |
| --- | --- | --- |
| The fused kernels | 1.25 times | 1.07 times |

Both together give 2.17 times. The direct measurements agree with the compiled
value: the fused kernels give 1.057 times at the pre-training shape and 1.044
times at the fine-tuning shape.

**The two changes do not add.** The fused kernels give 1.25 times without
compilation and 1.07 times with it. Inductor reaches most of what the
hand-written kernels reach. The baseline document reports 1.23 times for the
fused kernels and says that the value does not depend on the depth
distribution. That statement is correct, because both paths are read at the same
depths. The value does depend on `torch.compile`, and the baseline document does
not say so, because no measurement with `torch.compile` existed.

The order of the two changes therefore decides how each one looks. From the
eager PyTorch path, the fused kernels give 1.25 times and `torch.compile` gives
2.04 times. From the compiled PyTorch path, the fused kernels give only 1.07
times.

### Both phases now fit on an 80 GB card

This is the more useful result. The baseline document had to extrapolate,
because neither phase's shape fits on the available card:

| Phase | Shape | Eager (estimate) | Compiled (measured) |
| --- | --- | --- | --- |
| Pre-training | depth 256, crop 312 | 82.7 GB | 56.0 GB |
| Fine-tuning | depth 320, crop 320 | 99.5 GB | 66.4 GB |

Both compiled figures are below 80 GB, and both come from a run at that exact
shape. Section 5 of the baseline document records that fine-tuning does not fit
even on the 96 GB card that we infer for the paper. With `torch.compile`, it
fits on a card that is 16 GB smaller.

### The direct measurement, and what it says about the extrapolation

Both shapes therefore ran directly, at 5 timed steps each, with the paper's
effective batch for the phase: 12 for pre-training and 32 for fine-tuning.

Data:
[bench/results/h100-measured-pretrain.json](../bench/results/h100-measured-pretrain.json)
and
[bench/results/h100-measured-finetune.json](../bench/results/h100-measured-finetune.json).

| Shape | Variant | Estimate | Measurement | Difference |
| --- | --- | --- | --- | --- |
| Pre-training | `vanilla+compile` | 382 ms | 370 ms | −3.2 % |
| Pre-training | `cuequivariance+compile` | 359 ms | 350 ms | −2.6 % |
| Fine-tuning | `vanilla+compile` | 439 ms | 407 ms | −7.3 % |
| Fine-tuning | `cuequivariance+compile` | 414 ms | 390 ms | −5.8 % |

| Shape | Variant | Estimate | Measurement | Difference |
| --- | --- | --- | --- | --- |
| Pre-training | `vanilla+compile` | 55.9 GB | 56.0 GB | +0.2 % |
| Pre-training | `cuequivariance+compile` | 57.0 GB | 57.1 GB | +0.2 % |
| Fine-tuning | `vanilla+compile` | 67.0 GB | 66.4 GB | −0.9 % |
| Fine-tuning | `cuequivariance+compile` | 68.3 GB | 67.5 GB | −1.2 % |

Three results:

- **The memory fit is very accurate.** It predicts the direct measurement to
  1.2 per cent or better, across a reach of 1.43 times in MSA depth and a crop
  change that it never measured.
- **The time estimates are all high.** The extrapolation is thus conservative.
  It never claims a throughput that the hardware does not give.
- **The crop rescale over-corrects.** The pre-training estimates are 3 per cent
  high, and the fine-tuning estimates are 6 to 7 per cent high. Pre-training
  uses the crop of the sweep, and fine-tuning does not. The extra 3 to 4 per
  cent is therefore the cost of the arithmetic rescale from crop 312 to crop
  320. The pair term uses crop³, and that exponent is a little too large.

This is the first check of the extrapolation method against a measurement. The
baseline document has no such check, because no shape it extrapolates to fits on
the card. The agreement to 3 per cent for the same crop supports the numbers in
that document as well.

## 6. What this does not measure

- **The depth distribution of the training data.** All estimates use the depth
  caps. Real alignments are frequently smaller. This limit stays the same as in the
  baseline document, and it still controls the comparison with the paper.
- **Convergence.** Every number here is a time and a memory value. The probe
  shows that the compiled path computes the same loss and the same gradients to
  the bf16 floor, at one step. It does not show that a run of 68,000 steps
  reaches the same result. No compiled run has trained anything.
- **`max-autotune`.** All measurements use the `default` mode. `max-autotune`
  searches across kernel configurations and needs a much longer compilation. It
  is not measured.
- **The data pipeline, more than one GPU, and real data.** These exclusions stay the
  same as in the baseline document.
- **The crop axis.** The fine-tuning figures rescale a fit at crop 312 to crop
  320 by arithmetic, exactly as the baseline document does.

## 7. How to repeat the measurement

The probe first. It costs approximately two minutes of GPU time:

```bash
uv run modal run bench/modal_app.py::probe
```

Then the four-variant sweep:

```bash
uv run modal run bench/modal_app.py::sweep --variants vanilla,cuequivariance,vanilla+compile,cuequivariance+compile
```

Then each phase at its own shape. `--phase` sets the MSA depth, the crop and the
effective batch together, and labels the projection with the same phase:

```bash
uv run modal run bench/modal_app.py::main --phase pretrain --variants vanilla+compile,cuequivariance+compile --no-verify-first
```

```bash
uv run modal run bench/modal_app.py::main --phase finetune --variants vanilla+compile,cuequivariance+compile --no-verify-first
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
| [bench/results/h100-measured-pretrain.json](../bench/results/h100-measured-pretrain.json) | The compiled variants at depth 256, crop 312, effective batch 12 |
| [bench/results/h100-measured-finetune.json](../bench/results/h100-measured-finetune.json) | The compiled variants at depth 320, crop 320, effective batch 32 |

The fine-tuning file records a dirty working tree. The difference was in `docs/`
only. `bench/` was at commit `3b8d275` for both measured runs, and no commit
after it changes `bench/`.
