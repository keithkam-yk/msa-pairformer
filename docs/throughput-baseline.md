# Training throughput: the first baseline

This document records the first throughput baseline for MSA Pairformer training.
It gives the measurements, the changes that made the measurements possible, and
the extrapolation to the shapes in the paper.

The text uses ASD-STE100 Simplified Technical English. Sentences are short. Each
sentence gives one idea. The tenses are simple.

Date of the measurement: 12 August 2026. Repository commit: `617a94d`.
Measured data: [bench/results/h100-sweep.json](../bench/results/h100-sweep.json).
Analysis: [bench/results/h100-whole-run.json](../bench/results/h100-whole-run.json).

## Technical names

Each name below has one definition in this document. The document does not use
other words for the same thing.

| Name | Definition |
| --- | --- |
| alignment | One MSA. One forward pass and one backward pass process one alignment |
| MSA depth | The number of sequences (S) in one alignment |
| crop | The number of residues (N) in one alignment |
| depth cap | The maximum MSA depth that `hhfilter` permits in a training phase |
| depth sweep | A set of measurements at different MSA depths, with all other settings equal |
| PyTorch path | The triangle operations in plain PyTorch (`vanilla` in the data files) |
| cuEquivariance path | The same operations in the fused cuEquivariance kernels |
| activation checkpointing | Recompute of the triangle activations in the backward pass, in place of storage |
| pre-training | The paper's first training phase, with uniform sequence weights |
| fine-tuning | The paper's second training phase, with the query-biased outer product |
| the paper | Akiyama et al., Cell 2026 |

## 1. Summary

The paper trains in two phases and reports 10.5 days on one H100 for both of
them together. In that time it processes 1,176,000 alignments, which is 771 ms
for each alignment.

Our estimate of the same run, at each phase's depth cap:

| Path | ms for each alignment | Estimated days | Compared with the paper |
| --- | --- | --- | --- |
| PyTorch path | 822 | 11.18 | 0.94 times |
| cuEquivariance path | 667 | 9.07 | 1.16 times |

The paper's throughput is thus between our two paths. Model compute accounts for
almost all of the reported 10.5 days. There is no large unaccounted cost.

Read the estimate with two limits in mind:

- It uses each phase's depth cap. Real alignments are frequently smaller than
  the cap, so the true cost for each alignment is lower than these numbers.
  Section 7 gives the consequence.
- It contains model compute only. Section 3 lists the exclusions.

An earlier version of this document reported that approximately half of the
paper's step time was not model compute. That result was wrong. Section 8 gives
the cause.

## 2. What the paper reports

Training has two phases with different shapes and different batch sizes.

| Item | Pre-training | Fine-tuning |
| --- | --- | --- |
| Optimizer steps | 50,000 | 18,000 |
| Effective batch | 12 | 32 |
| Alignments | 600,000 | 576,000 |
| Depth cap | 256 | 320 |
| Crop | 312 | 320 |
| Minimum sequences | 8 | 128 |
| Masked positions | 15 % of all rows | 15 % of the query row |

Hardware and duration: "a single H100", 10.5 days for both phases.

The comparable unit is the alignment, not the optimizer step. The effective
batch changes from 12 to 32, so one step of fine-tuning is 2.7 times the work of
one step of pre-training. To add 50,000 steps to 18,000 steps is thus to add two
different quantities.

The count of alignments is `steps × effective batch`:

```
50,000 × 12 = 600,000        pre-training
18,000 × 32 = 576,000        fine-tuning
              ---------
              1,176,000      alignments in 10.5 days = 907,200 s
                             = 771 ms for each alignment
```

The micro-batch size does not change this count, and the paper does not give it.
The product is the same for each split of the effective batch. The split is also
forced: one alignment at depth 256 and crop 312 needs approximately 83 GB
(section 5), so no H100 holds two of them.

The paper does not name the memory size of the training card. A different
section of the paper describes a 96 GB NVIDIA H100 for the pairing experiments.
We infer that the training card is the same 96 GB part. The inference is not
necessary for the result, but section 5 shows that it does not fully explain the
memory.

The published repository contains no training code. There is no backward pass
and no optimizer step outside `pairing_optimization/`. The harness in `bench/`
is therefore a reconstruction. It is the smallest training step that is faithful
enough for a comparison.

## 3. What the harness measures, and what it excludes

The harness measures one optimizer step. The step contains:

- A masked-language-model loss across the full MSA.
- bf16 autocast for the forward pass and the backward pass.
- Gradient accumulation, with a micro-batch size of 1.
- An AdamW weight update.
- Activation checkpointing of the triangle updates.

The harness excludes these items. Each one adds time to a real training run:

- The data pipeline. The batches are synthetic.
- `torch.compile`. All code in this document runs eagerly.
  [throughput-torch-compile.md](throughput-torch-compile.md) measures it
  separately, and it changes both the time and the memory.
- Optimizer-state offload, and all other memory optimizations.
- Distributed communication. All measurements use one GPU.
- Checkpoint writes, logging and validation.

Method for the time measurement: 2 untimed warmup steps, then 5 timed steps.
The harness reports the median. It calls `torch.cuda.synchronize()` before and
after each step. It then divides by the accumulation count to give the time for
one alignment, which is the unit that both phases share.

## 4. Test hardware and software

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (SXM), one card |
| Platform | Modal container, Linux, gVisor |
| Python | 3.11.12 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| cuEquivariance ops | 0.11.1 (cu13 wheels) |
| Model parameters | 111,365,468 |
| float32 matmul precision | `highest` (TF32 is off) |

TF32 stays off on purpose. TF32 moves fp32 matmul results by approximately
1e-3 relative. That error is large enough to look like a numerical fault.

The card is an SXM part. The paper does not name its variant. A PCIe part is
slower than an SXM part. This is a caveat on every comparison in this document.

## 5. The memory limit

Neither phase's shape runs on an 80 GB H100. Peak allocated memory for one
alignment, at crop 312 and micro-batch 1:

| MSA depth | Checkpointing on (GB) | Checkpointing off (GB) |
| --- | --- | --- |
| 32 | 37.17 | 64.06 |
| 64 | 43.54 | 70.41 |
| 96 | 49.90 | 76.77 |
| 128 | 56.24 | out of memory |
| 192 | 68.83 | out of memory |
| 256 | out of memory | out of memory |

Both series increase at 0.198 GB for each sequence. Activation checkpointing
does not change this slope. The slope is the MSA track, and the checkpoints are
on the pair track. Checkpointing moves the intercept from 57.71 GB to 30.84 GB.
It thus saves a constant 27 GB.

Two independent runs agree on the memory. The third row is an earlier fit across
a subset of the same points as the second row. It is not independent evidence,
and the document shows it only for completeness.

| Source | Slope (GB/sequence) | Intercept (GB) | At depth 256 | At depth 320 |
| --- | --- | --- | --- | --- |
| Memory probe, no optimizer state | 0.1979 | 30.84 | 81.5 GB | 94.2 GB |
| Six-point depth sweep | 0.1975 | 32.17 | 82.7 GB | 95.4 GB |
| First depth sweep, 3 of the same depths | 0.1972 | 32.20 | 82.7 GB | 95.3 GB |

The probe does not allocate optimizer state. The sweeps do. The difference of
approximately 1.4 GB is that state. The two runs are separate: different code,
different container, different day. Their slopes agree to 0.2 per cent.

The consequence for each phase:

| Phase | Shape | Memory | Fits in 80 GB | Fits in 96 GB |
| --- | --- | --- | --- | --- |
| Pre-training | depth 256, crop 312 | 83 GB | no | yes |
| Fine-tuning | depth 320, crop 320 | 99 GB | no | **no** |

The fine-tuning figure applies a crop correction to the measured fit, and the
crop axis is not measured. Section 7 describes the correction. Even so, the
margin is 3 GB above a 96 GB card, and the direction is not in doubt.

The fine-tuning phase therefore does not fit on the inferred card with the
released code. Something that is not in the repository made it fit. The code
still contains chunk paths and layer-level checkpoint options. No code connects
them to a training loop.

We keep the 80 GB H100 for the throughput measurement. A larger card removes the
memory limit but adds a new difference. An H200 moves approximately 4.8 TB/s.
An H100 SXM moves approximately 3.35 TB/s. This workload is bandwidth-bound.
A measurement on an H200 is thus a measurement of a different memory system.

## 6. The depth sweep

All points come from one container, at one commit, on one physical GPU. Crop
312, gradient accumulation 12, bf16, activation checkpointing on.

| MSA depth | PyTorch path (s/step) | ms/alignment | cuEquivariance path (s/step) | ms/alignment | Peak memory (GB) |
| --- | --- | --- | --- | --- | --- |
| 32 | 6.427 | 536 | 4.357 | 363 | 38.5 |
| 64 | 6.732 | 561 | 4.823 | 402 | 44.8 |
| 96 | 7.198 | 600 | 5.309 | 442 | 51.1 |
| 128 | 7.547 | 629 | 5.668 | 472 | 57.5 |
| 160 | 7.962 | 664 | 6.063 | 505 | 63.8 |
| 192 | 8.425 | 702 | 6.524 | 544 | 70.1 |
| 224 | out of memory | – | out of memory | – | – |

The cuEquivariance path is faster at every depth. The advantage decreases as the
MSA depth increases: 1.48 times at depth 32, and 1.29 times at depth 192. The
structure of the model explains this. The fused kernels act on the pair track.
The cost of the pair track is O(crop³), which is constant in MSA depth. The MSA
track increases around it.

## 7. The extrapolation to the two phases

The cost is close to affine in MSA depth. The structure predicts this shape:

- The MSA track is O(depth × crop × d). It is linear in MSA depth.
- The pair track is O(crop³). It is constant in MSA depth.

The memory sweep in section 5 found the same shape. We fit a straight line to
the six measured points.

| Path | Fit (s/step) | r² | Adjusted r² | Largest residual (s) |
| --- | --- | --- | --- | --- |
| PyTorch path | 0.012525 × depth + 5.979 | 0.99709 | 0.99515 | 0.048 |
| cuEquivariance path | 0.013313 × depth + 3.966 | 0.99793 | 0.99655 | 0.065 |

Six points against two parameters leave four residual degrees of freedom.
Adjusted r² is therefore defined. An accidental repeat run measured the
run-to-run variation: 0.4 per cent to 2.7 per cent. The residuals are of the
same size as that noise. The line is therefore as good as the measurements
permit.

The two terms of the fit are the two tracks, so the document rescales them
separately for fine-tuning's crop of 320:

- The intercept is the pair track. It is O(crop³), thus × (320/312)³.
- The slope times depth is the MSA track. It is linear in crop, thus × 320/312.

That rescale is arithmetic. The sweep measures only the depth axis. The
rescale moves the fine-tuning estimate by approximately 6 per cent.

The result for the whole run:

| Phase | Shape | PyTorch path | cuEquivariance path |
| --- | --- | --- | --- |
| Pre-training, 600,000 alignments | depth 256, crop 312 | 765 ms → 5.32 days | 615 ms → 4.27 days |
| Fine-tuning, 576,000 alignments | depth 320, crop 320 | 880 ms → 5.87 days | 721 ms → 4.80 days |
| **Whole run, 1,176,000 alignments** | | **822 ms → 11.18 days** | **667 ms → 9.07 days** |
| The paper | | 771 ms → 10.5 days | |

Two sources of doubt sit on those totals. The parabola through the same six
points disagrees with the line by 2 per cent at depth 256 and by 4 per cent at
depth 320, which is the part of the estimate that is model choice and not
measurement. The crop rescale adds the 6 per cent described above. The
extrapolation reaches 1.33 times beyond the largest measured depth for
pre-training, and 1.67 times for fine-tuning.

The depth cap is the more important limit. Pre-training accepts alignments with
8 sequences or more, and fine-tuning accepts 128 or more. The typical alignment
is therefore smaller than the cap, and cheaper. Our totals are upper bounds on
model compute. The paper's 771 ms is a true average across the real depth
distribution. To compare the two exactly, that distribution must be known.

What the comparison supports:

- The paper's 10.5 days are consistent with model compute alone. The PyTorch
  path alone accounts for more than 100 per cent of the reported time at the
  caps, and the cuEquivariance path accounts for 86 per cent.
- The fused kernels give 1.23 times across the whole run, from 11.18 days to
  9.07 days. This does not depend on the depth distribution, because both paths
  are read at the same depths. It does depend on `torch.compile`: with
  compilation on, the same comparison gives only 1.07 times. See
  [throughput-torch-compile.md](throughput-torch-compile.md).

What the comparison does not support: a claim about how much of the paper's time
is data loading or other work outside the model. The depth distribution controls
that number, and we do not have it.

## 8. Changes that the measurement made necessary

Three tests passed, but they checked nothing. Only a rented GPU found them.

| Problem | Cause | Fix |
| --- | --- | --- |
| The baseline compared against the wrong quantity | We attributed all 10.5 days to the 50,000 pre-training steps. That gives 18.1 s for each step. The document then compared it with one shape, and reported that half of the paper's time was not model compute. Training has two phases, and the effective batch is not the same in both. | The comparison unit is now the alignment: `steps × effective batch` in both phases. `bench/config.py` holds a `Phase` for each. |
| The default configuration described neither phase | `BenchConfig` used depth 320 with crop 312 and effective batch 12. Depth 320 is fine-tuning. Crop 312 and batch 12 are pre-training. | `BenchConfig.for_phase()` fills depth, crop and effective batch from one `Phase`. |
| A test compared cuEquivariance with itself | `use_cuequivariance: bool = CUEQUIVARIANCE_AVAILABLE` is a default argument. Python evaluates it once, at import. On CUDA it froze to `True` for both sides of the comparison. | The test factories take the flag as a parameter. A guard asserts that the two outputs are not bitwise identical. |
| The fused kernels never ran in the golden fixtures | cuEquivariance declines small shapes. The fixtures used N = 10 and N = 24. The fused path returned the fallback result, and gave no warning. | The kernel test uses N = 312. A comment records the limit of the fixtures. |
| A parameter checksum failed in the container | The model computes some parameters. It does not draw them at random. `q_proj.weight + randn_like(...) * 0.1` differs by 1 ULP between AVX-512 and NEON. A SHA-256 checksum cannot tolerate that. | `param_fingerprint` records shape, absolute sum, square sum, minimum and maximum. It compares with rtol 1e-5. We recorded the goldens again. |
| A tolerance was too tight | The CPU tolerance was 1e-6, because the machine that recorded the goldens measured 0.0. That value is a property of one machine. | Each execution path has its own tolerance, set from measured deviation: 1e-4 relative, 5e-5 absolute. |
| The container held two CUDA stacks | PyTorch 2.13 is a cu13 build. The image installed cu12 operation wheels. | The image pins `torch==2.13.0` and installs cu13 wheels. `uv pip compile` found this before we rented any GPU. |
| The paper configuration stopped with an out-of-memory error | Activation checkpointing was off. The model default is `use_checkpointing_triangles=False`. | `BenchConfig.checkpoint_triangles` defaults to `True`. The run of 10.5 days did not use the unchecked path. |
| We read the benefit of the checkpoints incorrectly | We compared two allocations that were both near the memory limit. The apparent decrease was 1.3 GB. | The memory sweep showed a 27 GB intercept shift, and no slope change. |
| An r² gate that three points could not fail | Two of five requested depths caused an out-of-memory error. Three points against two parameters leave one residual degree of freedom. Adjusted r² is undefined there, and r² is near 1 for almost any three points. | `fit_line` reports `n_points`, `residual_dof` and `adjusted_r_squared`. `curvature_bracket` adds the parabola. A quotable estimate needs 5 points and r² ≥ 0.99. |
| A rerun measured the same three depths | The depth ladder existed twice. `DEFAULT_DEPTHS` grew to six points. The Modal entrypoint kept its own hardcoded string. | `resolve_depths("")` is the only source. A test asserts that the default ladder can satisfy the quote gate. |
| A silent fallback was possible | A container without the cuEquivariance wheels measures the PyTorch path under the cuEquivariance label. | `bench.train.run` raises an error before Modal bills any GPU time. |
| `modal run bench/modal_app.py` failed | Modal infers an entrypoint only when the file defines one. This file defines three. | The documentation gives the explicit `::entrypoint` form. |

Two changes to the project were also necessary:

- The project uses uv. `uv.lock` fixes the versions. The golden fixtures depend
  on an exact PyTorch build, so an unpinned version breaks the correctness suite.
- The Modal image installs with `uv_pip_install`, and the GPU is a flag.

The first row is the important one. The measurements did not change. Every
number in section 6 is the same as before. Only the quantity we compared them
with was wrong, and it was wrong in a direction that flattered us.

## 9. What we do not know yet

- The depth distribution of the training data. This now controls the main
  result, because it is the difference between the paper's true average cost
  for each alignment and our estimate at the caps. The dataset is the filtered
  OpenProteinSet UniClust30 set of 269,455 MSAs. The distribution is
  measurable.
- The crop axis. The fine-tuning estimate rescales the fit from crop 312 to crop
  320 by arithmetic. A sweep at crop 320 replaces that with a measurement:

  ```bash
  uv run modal run bench/modal_app.py::sweep --phase finetune
  ```

- The data pipeline is not measured. `hhfilter_select` starts a subprocess for
  each example, inside `__getitem__`. It remains the first component to profile,
  and the component that binds hardest on a node with 8 GPUs.
- The tolerances in `tests/test_cuequivariance.py` are still estimates (1e-2 and
  1e-3). The tests print the real deviations, but nobody has set the values from
  the print.
- `pyproject.toml` still pins `cuequivariance_ops_cu12` for Linux installs. That
  is the same mixed-stack fault as in section 8, and it ships to users.
- No measurement in this document uses more than one GPU or real data.

`torch.compile` was on this list. It is not any more:
[throughput-torch-compile.md](throughput-torch-compile.md) measures it. That
document supersedes two statements here. Compilation moves the memory below 80
GB for both phases, so the shapes in section 5 no longer need an extrapolation.
The direct measurement of the pre-training shape agrees with the extrapolation
in section 7 to 3 per cent on time and to 0.2 per cent on memory, which is the
best available check on the method of this document. Every number in this
document stays correct for the eager paths.

## 10. How to repeat the measurement

Correctness and drift first. This step establishes that the GPU paths work:

```bash
uv run modal run bench/modal_app.py::check
```

Then the baseline. This step runs the full depth sweep in one container:

```bash
uv run modal run bench/modal_app.py::sweep
```

The local self-check needs no GPU:

```bash
uv run pytest
```

## 11. Data files

| File | Content |
| --- | --- |
| [bench/results/h100-sweep.json](../bench/results/h100-sweep.json) | The six-point depth sweep, as measured |
| [bench/results/h100-whole-run.json](../bench/results/h100-whole-run.json) | The two-phase analysis derived from those points |
| [bench/results/h100-memory-depth.json](../bench/results/h100-memory-depth.json) | The memory probe, with and without activation checkpointing |
| [bench/results/h100-drift.json](../bench/results/h100-drift.json) | The deviation of each execution path from the golden fixtures |

The sweep file also holds `paper_estimate` fields from before the two-phase
correction. They target depth 320 with crop 312 and effective batch 12, which
is one value from each phase. The measured points in that file are correct. Use
the whole-run file for the analysis.

Each result file records the commit, the environment, the precision settings and
the full configuration. A timing number without that record is not evidence.
