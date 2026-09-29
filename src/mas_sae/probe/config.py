"""Reusable config dataclasses for probe-related scripts, loadable from YAML.

Moved out of scripts/activation_layer_selection_probe.py per review feedback,
so the config can be imported and reused elsewhere rather than living only in
one script.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ActivationLayerSelectionConfig:
    seed: int = 42
    run_name: str = "scale_smoke_1q"
    candidate_layers: list = field(default_factory=lambda: ["08", "17", "25", "33"])
    data_root: Path = Path("data/activations")
    results_root: Path = Path("results/collection")
    output_dir: Path = Path("results/activation_layer_selection")

    @classmethod
    def from_yaml(cls, path: Path) -> "ActivationLayerSelectionConfig":
        """Load config values from a YAML file, falling back to the
        dataclass defaults for anything not specified. Path fields in the
        YAML are read as plain strings and converted to Path here."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        path_field_names = {
            f.name for f in fields(cls) if f.type in (Path, "Path")
        }
        for key in list(raw.keys()):
            if key in path_field_names:
                raw[key] = Path(raw[key])

        return cls(**raw)
