"""Shared run-versioning helper. Used by mas_sae.probe.layer_selection so
results from repeated runs never overwrite each other."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def create_versioned_run_dir(results_root: Path) -> Path:
    """Create a fresh, timestamped subdirectory under results_root for this
    run's outputs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = results_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
