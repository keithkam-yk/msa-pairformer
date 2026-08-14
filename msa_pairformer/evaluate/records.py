from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class UnknownPolicy(Enum):
    AS_NEGATIVE = "as_negative"
    EXCLUDE = "exclude"


# -- Projection ----------------------------------------------------------


class Projection:
    """Maps between model frame, structure frame, and scored frame. ADR-0001 D1."""

    def project_pred(self, pred: NDArray) -> NDArray:
        raise NotImplementedError

    def project_truth(self, truth: NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def chain_break(self) -> int | None:
        return None


@dataclass(frozen=True)
class MaskPairProjection(Projection):
    """Boolean mask pair: msa_mask (model frame) and cif_mask (structure frame)."""

    msa_mask: NDArray[np.bool_]
    cif_mask: NDArray[np.bool_]
    chain_break_model: int | None = None


@dataclass(frozen=True)
class IndexProjection(Projection):
    """Index list into model-frame columns; truth side is identity."""

    msa_subset_idx: NDArray[np.intp]


@dataclass(frozen=True)
class OffsetProjection(Projection):
    """Scalar offset between model-frame and structure-frame numbering."""

    offset: int


# -- Target records ------------------------------------------------------


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
    projection: OffsetProjection
    msa_file: str
