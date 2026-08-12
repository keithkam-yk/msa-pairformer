"""Experiment configuration as a validated value, not a loose dict.

A benchmark configuration travels a long way -- argparse or Modal's CLI, into
`run()`, across the `.remote()` boundary, and back out inside a JSON result that
gets committed and compared against later runs. A dict silently tolerates a
misspelled key, a string where an int belongs, and a field that quietly stopped
being read; each of those turns into a result that looks fine and means
something other than what it claims.

`BenchConfig` is frozen so a run cannot mutate the configuration it reports, and
validated at construction so a bad value fails before a GPU is billed rather
than inside the timing loop.

The paper constants live here because they are configuration too: they are the
defaults every run is measured against, and the projection is only meaningful
relative to them. The paper trains in two phases with different shapes and
different effective batches, so those constants are a pair of `Phase` values
rather than one flat set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, get_args


@dataclass(frozen=True)
class Phase:
    """One of the paper's two training phases.

    `depth` is the cap `hhfilter` enforces, not the typical alignment depth, so
    a measurement at `depth` is an upper bound on that phase's cost.
    """

    key: str
    steps: int
    effective_batch: int
    depth: int
    crop: int
    min_sequences: int

    @property
    def examples(self) -> int:
        """Alignments processed over the whole phase.

        One forward+backward each. The paper does not state how it split the
        effective batch into micro-batches, and it does not need to: the
        product is the same however it was split.
        """
        return self.steps * self.effective_batch


# Akiyama et al., Cell 2026. Training runs in two phases and the reported
# 10.5 days on a single H100 covers both of them.
PRETRAIN = Phase("pretrain", steps=50_000, effective_batch=12, depth=256,
                 crop=312, min_sequences=8)
FINETUNE = Phase("finetune", steps=18_000, effective_batch=32, depth=320,
                 crop=320, min_sequences=128)
PHASES: dict[str, Phase] = {p.key: p for p in (PRETRAIN, FINETUNE)}

REPORTED_DAYS: float = 10.5
TOTAL_EXAMPLES: int = PRETRAIN.examples + FINETUNE.examples

# The unit that is comparable across the two phases. An optimizer step is not:
# the effective batch goes from 12 to 32, so a step in fine-tuning is 2.7 times
# the work of a step in pre-training, and 68,000 "steps" adds two different
# things together. One alignment's forward+backward is the same unit in both.
#
# Micro-batch size does not enter this. steps x effective_batch counts the
# alignments however the accumulation was split, and at these shapes the split
# is forced anyway: one example at depth 256 and crop 312 already needs ~83 GB,
# so micro_batch is 1 on any H100.
PAPER_SECONDS_PER_EXAMPLE: float = (
    REPORTED_DAYS * 86_400 / TOTAL_EXAMPLES
)

Amp = Literal["bf16", "fp32"]
Precision = Literal["highest", "high", "medium"]


@dataclass(frozen=True)
class BenchConfig:
    """One measurable configuration.

    Defaults reproduce the paper's reported setup, so a bare `BenchConfig()` is
    the comparison case rather than an arbitrary starting point.
    """

    device: str = "cuda"
    # Pre-training, because that is the phase the depth sweep measures. The
    # previous defaults -- depth 320 with crop 312 and accum 12 -- were one
    # value from each phase and described neither.
    phase: str = PRETRAIN.key
    depth: int = PRETRAIN.depth
    crop: int = PRETRAIN.crop
    micro_batch: int = 1
    accum: int = PRETRAIN.effective_batch
    steps: int = 5
    warmup: int = 2
    amp: Amp = "bf16"
    lr: float = 1e-4
    cuequivariance: bool = True
    # Defaults on, unlike the model's own `use_checkpointing_triangles=False`,
    # because at either phase's shape it is not optional: depth 320 with a 312
    # crop allocates 76.2 GiB and dies on an 80 GB H100 without it, at
    # micro_batch 1 with nothing left to shrink. Whatever the authors ran for
    # 10.5 days, it was not the unchecked path, and the flag exists in
    # PairwiseBlock precisely for this.
    checkpoint_triangles: bool = True
    # Pinned rather than inherited: torch has moved this default across
    # versions, and TF32 alone drifts fp32 matmuls by ~1e-3 relative, which is
    # large enough to be mistaken for a real numerical regression.
    float32_precision: Precision = "highest"

    def __post_init__(self) -> None:
        positive = ("depth", "crop", "micro_batch", "accum", "steps")
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {self.warmup}")
        if self.amp not in get_args(Amp):
            raise ValueError(f"amp must be one of {get_args(Amp)}, got {self.amp!r}")
        if self.float32_precision not in get_args(Precision):
            raise ValueError(
                f"float32_precision must be one of {get_args(Precision)}, "
                f"got {self.float32_precision!r}"
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.phase not in PHASES:
            raise ValueError(
                f"phase must be one of {sorted(PHASES)}, got {self.phase!r}"
            )

    @property
    def paper_phase(self) -> Phase:
        """The phase this run is compared against."""
        return PHASES[self.phase]

    @classmethod
    def for_phase(cls, phase: str, **overrides: Any) -> BenchConfig:
        """Build a config from a paper phase, so shape and comparison agree.

        Passing depth, crop and accum by hand is how the old defaults came to
        mix the two phases.
        """
        spec = PHASES[phase]
        return cls(phase=spec.key, depth=spec.depth, crop=spec.crop,
                   accum=spec.effective_batch, **overrides)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.accum

    @property
    def tokens_per_micro_batch(self) -> int:
        return self.micro_batch * self.depth * self.crop

    def to_dict(self) -> dict[str, Any]:
        """Wire and JSON form. Used at the `.remote()` boundary so the result
        shape is identical whether a run was local or remote."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BenchConfig:
        """Reconstruct, rejecting unknown keys.

        A silently-ignored key is the failure this type exists to prevent: a
        run configured with `--micro-batch` reaching a field named
        `micro_batches` would measure the default and report the request.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown config keys: {sorted(unknown)} (known: {sorted(known)})"
            )
        return cls(**raw)
