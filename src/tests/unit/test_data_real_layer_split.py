"""Unit tests for mas_sae.probe.data.load_real_layer_split -- addresses
review feedback requesting test coverage for data.py's real-data loading
path (previously only the synthetic generators were tested)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mas_sae.probe.data import load_real_layer_split


@pytest.fixture
def fake_collection_run(tmp_path):
    """Builds a minimal fake collection run on disk, matching the real
    schema: data/activations/<run_name>/layer_<N>/{split}_attempt2.pt and
    results/collection/<run_name>/<split>/interactions.jsonl."""
    run_name = "unit_test_run"
    split = "train"
    layer = "08"

    data_root = tmp_path / "data" / "activations"
    results_root = tmp_path / "results" / "collection"

    layer_dir = data_root / run_name / f"layer_{layer}"
    layer_dir.mkdir(parents=True)

    # 3 fake activation rows, 4-dim for speed (real ones are 2560-dim)
    activations = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    torch.save(activations, layer_dir / f"{split}_attempt2.pt")

    interactions_dir = results_root / run_name / split
    interactions_dir.mkdir(parents=True)

    records = [
        {"attempt2_activation_index": 0, "solver_accepted_feedback": True, "critic_noncommittal": False},
        {"attempt2_activation_index": 1, "solver_accepted_feedback": False, "critic_noncommittal": False},
        {"attempt2_activation_index": 2, "solver_accepted_feedback": True, "critic_noncommittal": True},
    ]
    with open(interactions_dir / "interactions.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return {
        "run_name": run_name,
        "split": split,
        "layer": layer,
        "data_root": data_root,
        "results_root": results_root,
        "activations": activations,
    }


def test_load_real_layer_split_returns_expected_shapes(fake_collection_run):
    X, y = load_real_layer_split(
        fake_collection_run["run_name"],
        fake_collection_run["split"],
        fake_collection_run["layer"],
        fake_collection_run["data_root"],
        fake_collection_run["results_root"],
    )
    # 3 records total, but 1 is noncommittal and must be excluded
    assert X.shape == (2, 4)
    assert y.shape == (2,)


def test_load_real_layer_split_excludes_noncommittal_episodes(fake_collection_run):
    X, y = load_real_layer_split(
        fake_collection_run["run_name"],
        fake_collection_run["split"],
        fake_collection_run["layer"],
        fake_collection_run["data_root"],
        fake_collection_run["results_root"],
    )
    # the noncommittal record (index 2) must not appear anywhere in the output
    activations_np = fake_collection_run["activations"].numpy()
    assert not any(np.array_equal(row, activations_np[2]) for row in X)


def test_load_real_layer_split_aligns_activations_to_correct_labels(fake_collection_run):
    X, y = load_real_layer_split(
        fake_collection_run["run_name"],
        fake_collection_run["split"],
        fake_collection_run["layer"],
        fake_collection_run["data_root"],
        fake_collection_run["results_root"],
    )
    activations_np = fake_collection_run["activations"].numpy()

    # record 0 (index 0, label True) and record 1 (index 1, label False)
    # must map to the correct rows and the correct labels, in order
    assert np.array_equal(X[0], activations_np[0])
    assert y[0] == 1
    assert np.array_equal(X[1], activations_np[1])
    assert y[1] == 0


def test_load_real_layer_split_returns_binary_labels(fake_collection_run):
    _, y = load_real_layer_split(
        fake_collection_run["run_name"],
        fake_collection_run["split"],
        fake_collection_run["layer"],
        fake_collection_run["data_root"],
        fake_collection_run["results_root"],
    )
    assert set(np.unique(y)).issubset({0, 1})
