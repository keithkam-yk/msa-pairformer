"""Record what produced a measurement.

A timing number without the code, machine and configuration that produced it is
not evidence -- it cannot be compared against a later run or reproduced. Both
functions here degrade to nulls rather than raising, so a missing `git` or an
odd platform costs metadata and not the run.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

import torch


def git_info() -> dict[str, Any]:
    """Repository state at the time of the run.

    Returns nulls when git is unavailable. That happens in the Modal image,
    which carries Python source but no `.git` -- `bench/modal_app.py` therefore
    captures this on the client and passes it through to the container.
    """
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5,
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD")
    return {
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit does not describe what actually ran.
        "dirty": bool(git("status", "--porcelain")) if commit else None,
    }


def environment() -> dict[str, Any]:
    """Everything about the machine and build that could move a timing number.

    Call this *before* any cuEquivariance override: `cuequivariance_available`
    is meant to describe the machine, not the variant being measured. Reading it
    afterwards would make an A/B of the two triangle paths look as though it had
    run on two different hosts.
    """
    from msa_pairformer import pairwise_operations

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuequivariance_available": pairwise_operations.CUEQUIVARIANCE_AVAILABLE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
