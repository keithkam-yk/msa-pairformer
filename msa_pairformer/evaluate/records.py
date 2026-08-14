from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class UnknownPolicy(Enum):
    AS_NEGATIVE = "as_negative"
    EXCLUDE = "exclude"


@dataclass(frozen=True)
class Projection:
    """Selects scored-frame positions from model-frame and structure-frame arrays.

    Adapters convert benchmark-specific representations (boolean masks,
    index lists) into index arrays at construction time.
    """

    pred_idx: NDArray[np.intp]
    truth_idx: NDArray[np.intp]
    chain_break: int | None = None


@dataclass(frozen=True)
class ContactTarget:
    target_id: str
    contacts: NDArray[np.float64]
    projection: Projection
    unknown_policy: UnknownPolicy
    msa_file: str


@dataclass(frozen=True)
class DMSTarget:
    target_id: str
    offset: int
    msa_file: str
