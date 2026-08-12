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
relative to them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, get_args

# Reported in Akiyama et al., Cell 2026: 50,000 optimizer steps on a single
# H100 in 10.5 days, effective batch 12, depth 320, 312-residue crop.
PAPER_STEPS: int = 50_000
PAPER_DAYS: float = 10.5
PAPER_SECONDS_PER_STEP: float = PAPER_DAYS * 86_400 / PAPER_STEPS
PAPER_DEPTH: int = 320
PAPER_CROP: int = 312
PAPER_ACCUM: int = 12

Amp = Literal["bf16", "fp32"]
Precision = Literal["highest", "high", "medium"]


@dataclass(frozen=True)
class BenchConfig:
    """One measurable configuration.

    Defaults reproduce the paper's reported setup, so a bare `BenchConfig()` is
    the comparison case rather than an arbitrary starting point.
    """

    device: str = "cuda"
    depth: int = PAPER_DEPTH
    crop: int = PAPER_CROP
    micro_batch: int = 1
    accum: int = PAPER_ACCUM
    steps: int = 5
    warmup: int = 2
    amp: Amp = "bf16"
    lr: float = 1e-4
    cuequivariance: bool = True
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
