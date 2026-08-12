# Training throughput: the first baseline

This document records the first throughput baseline for MSA Pairformer training.
It gives the measurements, the changes that made the measurements possible, and
the extrapolation to the MSA depth in the paper.

The text uses ASD-STE100 Simplified Technical English. Sentences are short. Each
sentence gives one idea. The tenses are simple.

Date of the measurement: 12 August 2026. Repository commit: `617a94d`.
Result data: [bench/results/h100-sweep.json](../bench/results/h100-sweep.json).

## Technical names

Each name below has one definition in this document. The document does not use
other words for the same thing.

| Name | Definition |
| --- | --- |
| MSA depth | The number of sequences (S) in one MSA batch element |
| crop | The number of residues (N) in one MSA batch element |
| depth sweep | A set of measurements at different MSA depths, with all other settings equal |
| PyTorch path | The triangle operations in plain PyTorch (`vanilla` in the data files) |
| cuEquivariance path | The same operations in the fused cuEquivariance kernels |
| activation checkpointing | Recompute of the triangle activations in the backward pass, in place of storage |
| gradient accumulation | Several micro-batches per optimizer step |
| optimizer step | One forward pass, one backward pass and one weight update, for all 12 micro-batches |
| the paper | Akiyama et al., Cell 2026 |

## 1. Summary

The paper reports 18.1 s for each optimizer step. Our estimate at the same MSA
depth is 7.9 s to 10.4 s. The model compute is thus 1.7 to 2.3 times faster than
the full step in the paper.

| Path | Estimated s/step at depth 320 | Estimated days for 50,000 steps | Ratio to the paper |
| --- | --- | --- | --- |
| PyTorch path | 9.99 – 10.40 | 5.78 – 6.02 | 1.74 – 1.82 |
| cuEquivariance path | 7.95 – 8.23 | 4.60 – 4.76 | 2.21 – 2.28 |

Read the ratio with care. It does not show that we made training faster. The two
numbers measure different things:

- Our number contains model compute only. Section 3 lists the exclusions.
- The number in the paper contains the full training step. It includes the data
  pipeline, the checkpoint writes and all other work.

The conclusion is therefore about the difference, not about the model. If the
model compute is near half of the step time in the paper, the other half is not
model compute. The data pipeline is the first candidate. Section 9 gives the
next steps.

## 2. What the paper reports

| Item | Value |
| --- | --- |
| Hardware | "a single H100" |
| Duration | 10.5 days |
| Optimizer steps | 50,000 |
| Seconds for each step | 18.144 (calculated) |
| MSA depth | 320 |
| Crop | 312 |
| Effective batch | 12 |

The paper does not name the memory size of the training card. A different
section of the paper describes a 96 GB NVIDIA H100 for the pairing experiments.
We infer that the training card is the same 96 GB part. Section 5 shows that the
inference agrees with three independent memory measurements. The inference is
not necessary for the result. It only explains why the reported configuration
does not run on our card.

The published repository contains no training code. There is no backward pass
and no optimizer step outside `pairing_optimization/`. The harness in `bench/`
is therefore a reconstruction. It is the smallest training step that is
faithful enough for a comparison.

## 3. What the harness measures, and what it excludes

The harness measures one optimizer step. The step contains:

- A masked-language-model loss across the full MSA.
- bf16 autocast for the forward pass and the backward pass.
- Gradient accumulation of 12 micro-batches, with a micro-batch size of 1.
- An AdamW weight update.
- Activation checkpointing of the triangle updates.

The harness excludes these items. Each one adds time to a real training run:

- The data pipeline. The batches are synthetic.
- `torch.compile`. All code runs eagerly.
- Optimizer-state offload, and all other memory optimizations.
- Distributed communication. All measurements use one GPU.
- Checkpoint writes, logging and validation.

Every parameter except the MSA depth holds the value in the paper. The MSA depth
is thus the only difference between the measurement and the comparison.

Method for the time measurement: 2 untimed warmup steps, then 5 timed steps.
The harness reports the median. It calls `torch.cuda.synchronize()` before and
after each step.

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

The configuration in the paper does not run on an 80 GB H100. Two attempts
stopped with an out-of-memory error. A memory sweep replaced a third attempt.

Peak allocated memory for one micro-step, at crop 312 and micro-batch 1:

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

Two independent runs agree on the memory at MSA depth 320. The third row is an
earlier fit across a subset of the same points as the second row. It is not
independent evidence, and the document shows it only for completeness.

| Source | Slope (GB/sequence) | Intercept (GB) | Value at depth 320 |
| --- | --- | --- | --- |
| Memory probe, no optimizer state | 0.1979 | 30.84 | 94.2 GB |
| Six-point depth sweep | 0.1975 | 32.17 | 95.4 GB |
| First depth sweep, 3 of the same depths | 0.1972 | 32.20 | 95.3 GB |

The probe does not allocate optimizer state. The sweeps do. The difference of
approximately 1.4 GB is that state. The two runs are separate: different code,
different container, different day. Their slopes agree to 0.2 per cent. This
agreement is the reason to trust the extrapolation in section 7.

The result is 94 GB to 95 GB at MSA depth 320. That value does not fit in 80 GB.
It fits in 96 GB. The largest MSA depth that fits on our card is between 192 and
224.

This limit is a property of the repository, not of the harness. The released
code is inference-shaped. The training script that made 10.5 days possible is
not in the repository. The code still contains chunk paths and layer-level
checkpoint options. No code connects them to a training loop.

We keep the 80 GB H100 for the throughput measurement. A larger card removes the
memory limit but adds a new difference. An H200 moves approximately 4.8 TB/s.
An H100 SXM moves approximately 3.35 TB/s. This workload is bandwidth-bound.
A measurement on an H200 is thus a measurement of a different memory system.

## 6. The depth sweep

All points come from one container, at one commit, on one physical GPU. Crop
312, gradient accumulation 12, bf16, activation checkpointing on.

| MSA depth | PyTorch path (s) | tokens/s | cuEquivariance path (s) | tokens/s | Peak memory (GB) |
| --- | --- | --- | --- | --- | --- |
| 32 | 6.427 | 18.6k | 4.357 | 27.5k | 38.5 |
| 64 | 6.732 | 35.6k | 4.823 | 49.7k | 44.8 |
| 96 | 7.198 | 49.9k | 5.309 | 67.7k | 51.1 |
| 128 | 7.547 | 63.5k | 5.668 | 84.6k | 57.5 |
| 160 | 7.962 | 75.2k | 6.063 | 98.8k | 63.8 |
| 192 | 8.425 | 85.3k | 6.524 | 110.2k | 70.1 |
| 224 | out of memory | – | out of memory | – | – |

The cuEquivariance path is faster at every depth. The advantage decreases as the
MSA depth increases: 1.48 times at depth 32, and 1.29 times at depth 192. The
structure of the model explains this. The fused kernels act on the pair track.
The cost of the pair track is O(crop³), which is constant in MSA depth. The MSA
track increases around it.

## 7. The extrapolation to MSA depth 320

The cost is close to affine in MSA depth. The structure predicts this shape:

- The MSA track is O(depth × crop × d). It is linear in MSA depth.
- The pair track is O(crop³). It is constant in MSA depth.

The memory sweep in section 5 found the same shape. We fit a straight line to
the six measured points and read the line at MSA depth 320.

| Path | Fit (s) | r² | Adjusted r² | Largest residual (s) |
| --- | --- | --- | --- | --- |
| PyTorch path | 0.012525 × depth + 5.979 | 0.99709 | 0.99515 | 0.048 |
| cuEquivariance path | 0.013313 × depth + 3.966 | 0.99793 | 0.99655 | 0.065 |

Six points against two parameters leave four residual degrees of freedom.
Adjusted r² is therefore defined. An accidental repeat run measured the
run-to-run variation: 0.4 per cent to 2.7 per cent, or 0.03 s to 0.13 s. The
residuals are of the same size as that noise. The line is therefore as good as
the measurements permit.

r² measures the fit inside the measured range. An extrapolation is outside that
range, where curvature decides the answer. A high r² hides mild curvature. We
therefore also fit a parabola to the same points, and we report both values.
The gap between them is the part of the estimate that is not evidence.

| Path | Straight line at 320 (s) | Parabola at 320 (s) | Spread |
| --- | --- | --- | --- |
| PyTorch path | 9.99 | 10.40 | 4.1 per cent |
| cuEquivariance path | 8.23 | 7.95 | 3.4 per cent |

The extrapolation reaches 1.67 times beyond the largest measured depth. The
estimate is not a measurement. The data files record this fact in the field
`is_measurement: false`.

An earlier sweep found three points and suggested curvature. That curvature was
noise. The parabola for the cuEquivariance path now bends in the opposite
direction: 7.95 s, against 8.96 s in the first sweep. A curvature that changes
sign between runs is not curvature. The parabola for the PyTorch path stays
above the line in both runs. We therefore keep its bracket open in the upward
direction. In both paths the spread decreased from 10.5 per cent to 3 to 4 per
cent.

## 8. Changes that the measurement made necessary

Three tests passed, but they checked nothing. Only a rented GPU found them.

| Problem | Cause | Fix |
| --- | --- | --- |
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

## 9. What we do not know yet

- The data pipeline is not measured. `hhfilter_select` starts a subprocess for
  each example, inside `__getitem__`. This is the first component to profile.
  It is also the component that binds hardest on a node with 8 GPUs, where one
  input pipeline feeds all of them.
- The tolerances in `tests/test_cuequivariance.py` are still estimates (1e-2 and
  1e-3). The tests print the real deviations, but nobody has set the values from
  the print.
- `pyproject.toml` still pins `cuequivariance_ops_cu12` for Linux installs. That
  is the same mixed-stack fault as in section 8, and it ships to users.
- No measurement uses `torch.compile`, more than one GPU, or real data.

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
| [bench/results/h100-sweep.json](../bench/results/h100-sweep.json) | The six-point depth sweep, the fits and the estimates |
| [bench/results/h100-memory-depth.json](../bench/results/h100-memory-depth.json) | The memory probe, with and without activation checkpointing |
| [bench/results/h100-drift.json](../bench/results/h100-drift.json) | The deviation of each execution path from the golden fixtures |

Each result file records the commit, the environment, the precision settings and
the full configuration. A timing number without that record is not evidence.
