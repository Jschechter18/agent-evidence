import random
import re
from typing import Any

import numpy as np
import torch

from mas_sae.experiments.artifacts import sha256_json


def config_sha256(config: dict[str, Any]) -> str:
    """Digest of every scientifically meaningful setting (all but ``output``).

    Covers roles, dataset revision and split, sampling, protocol and seed;
    ``output`` holds run names and operational settings such as chunk size,
    so changing those never blocks a resume.
    """
    return sha256_json({key: value for key, value in config.items() if key != "output"})


def validate_commit_revision(value: Any, name: str) -> None:
    """Require an immutable Hub revision: a full lowercase 40-hex commit SHA."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{40}", value) is None
    ):
        raise ValueError(
            f"{name}.revision must pin an immutable revision "
            "(a full lowercase 40-character hex commit SHA)."
        )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
