import json
from pathlib import Path

import pytest
import torch
import yaml

from mas_sae.data.activation_store import ActivationStore
from mas_sae.evaluation.behavior import classify_behavior
from mas_sae.experiments import collection_artifacts
from mas_sae.experiments.collection_artifacts import (
    save_collection_artifacts,
)
from mas_sae.experiments.records import read_jsonl


SITES = [
    "model.language_model.layers.8",
    "model.language_model.layers.17",
]


def test_save_collection_artifacts(tmp_path: Path) -> None:
    attempt1 = {
        site: [torch.ones(1, 4), torch.zeros(1, 4)]
        for site in SITES
    }
    attempt2 = {
        site: [
            torch.full((1, 4), float(index))
            for index in range(6)
        ]
        for site in SITES
    }

    records = [
        {
            "question_id": "q1" if index < 3 else "q2",
            "attempt1_activation_index": 0 if index < 3 else 1,
            "attempt2_activation_index": index,
            "critic_condition": "natural",
            "solver_accepted_feedback": [
                True, False, None, True, False, None
            ][index],
        }
        for index in range(6)
    ]

    config = {
        "model": {"id": "google/gemma-3-4b-it"},
        "dataset": {
            "source_split": "train",
            "num_questions": 2,
        },
        "collection": {"layers": [8, 17], "seed": 42},
        "output": {"run_name": "test_run"},
    }

    summary = save_collection_artifacts(
        activation_root=tmp_path / "activations",
        result_root=tmp_path / "results",
        run_name="test_run",
        source_split="train",
        candidate_sites=SITES,
        attempt1_by_site=attempt1,
        attempt2_by_site=attempt2,
        records=records,
        resolved_config=config,
    )

    store = ActivationStore(
        tmp_path / "activations" / "test_run" / "layer_08"
    )

    assert store.load_activations("train_attempt1").shape == (2, 4)
    assert store.load_activations("train_attempt2").shape == (6, 4)
    assert store.load_activations("train").shape == (8, 4)

    result_dir = tmp_path / "results" / "test_run" / "train"
    saved_records = read_jsonl(result_dir / "interactions.jsonl")

    assert [
        row["sae_attempt1_index"] for row in saved_records
    ] == [0, 0, 0, 1, 1, 1]

    assert [
        row["sae_attempt2_index"] for row in saved_records
    ] == [2, 3, 4, 5, 6, 7]

    with (result_dir / "resolved_config.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        assert yaml.safe_load(file) == config

    assert summary["num_questions"] == 2
    assert summary["num_episodes"] == 6
    assert summary["sae_rows_per_layer"] == 8
    assert summary["accepted"] == 2
    assert summary["rejected"] == 2
    assert summary["unlabeled_noncommittal"] == 2



def test_save_collection_artifacts_writes_sampled_questions(
    tmp_path: Path,
) -> None:
    attempt1 = {site: [torch.ones(1, 4), torch.zeros(1, 4)] for site in SITES}
    attempt2 = {
        site: [torch.full((1, 4), float(index)) for index in range(6)]
        for site in SITES
    }
    records = [
        {
            "question_id": "2hop__1_2" if index < 3 else "3hop1__3_4",
            "experiment_split": "discovery" if index < 3 else "validation",
            "attempt1_activation_index": 0 if index < 3 else 1,
            "attempt2_activation_index": index,
            "critic_condition": "natural",
            "solver_accepted_feedback": True,
        }
        for index in range(6)
    ]
    sampled = [
        {
            "position": 0,
            "question_id": "2hop__1_2",
            "hop_type": "2hop",
            "hop_group": "2hop",
            "experiment_split": "discovery",
        },
        {
            "position": 1,
            "question_id": "3hop1__3_4",
            "hop_type": "3hop1",
            "hop_group": "3hop",
            "experiment_split": "validation",
        },
    ]

    summary = save_collection_artifacts(
        activation_root=tmp_path / "activations",
        result_root=tmp_path / "results",
        run_name="v2_run",
        source_split="train",
        candidate_sites=SITES,
        attempt1_by_site=attempt1,
        attempt2_by_site=attempt2,
        records=records,
        resolved_config={"output": {"run_name": "v2_run"}},
        sampled_questions=sampled,
    )

    result_dir = tmp_path / "results" / "v2_run" / "train"
    with (result_dir / "sampled_questions.json").open(
        "r", encoding="utf-8"
    ) as file:
        assert json.load(file) == sampled

    assert summary["hop_group_counts"] == {"2hop": 1, "3hop": 1}
    assert summary["experiment_split_counts"] == {
        "discovery": 1,
        "validation": 1,
    }

    saved_records = read_jsonl(result_dir / "interactions.jsonl")
    assert [row["experiment_split"] for row in saved_records] == [
        "discovery"
    ] * 3 + ["validation"] * 3


def test_save_collection_artifacts_rejects_manifest_length_mismatch(
    tmp_path: Path,
) -> None:
    attempt1 = {site: [torch.ones(1, 4)] for site in SITES}
    attempt2 = {site: [torch.ones(1, 4)] * 3 for site in SITES}
    records = [
        {
            "question_id": "2hop__1_2",
            "attempt1_activation_index": 0,
            "attempt2_activation_index": index,
            "solver_accepted_feedback": True,
        }
        for index in range(3)
    ]

    with pytest.raises(ValueError, match="does not match"):
        save_collection_artifacts(
            activation_root=tmp_path / "activations",
            result_root=tmp_path / "results",
            run_name="bad",
            source_split="train",
            candidate_sites=SITES,
            attempt1_by_site=attempt1,
            attempt2_by_site=attempt2,
            records=records,
            resolved_config={},
            sampled_questions=[],
        )


def test_integrity_rejects_wrong_indices_before_writes(tmp_path):
    with pytest.raises(ValueError, match="indices"):
        save_collection_artifacts(
            activation_root=tmp_path / "activations", result_root=tmp_path / "results",
            run_name="bad", source_split="train", candidate_sites=SITES[:1],
            attempt1_by_site={SITES[0]: [torch.ones(1, 4)]},
            attempt2_by_site={SITES[0]: [torch.ones(1, 4)]},
            records=[{"question_id": "q", "attempt1_activation_index": 9, "attempt2_activation_index": 0}],
            resolved_config={})
    assert not (tmp_path / "activations").exists()
    assert not (tmp_path / "results").exists()


def test_production_summary_does_not_turn_invalid_feedback_into_rejection():
    row = {"solver_accepted_feedback": False,
           **classify_behavior("Paris", "Paris", "London", usable=False)}
    summary = collection_artifacts.behavioral_summary([row], production=True)
    assert summary["solver_behavior_counts"] == {"ambiguous": 1}
    assert summary["critic_position_counts"] == {"ambiguous_or_unresolved": 1}
    assert "rejected" not in summary
    assert summary["legacy_acceptance_counts"]["rejected"] == 1
    assert row["direct_critic_adoption"] is None
    assert collection_artifacts.behavioral_summary([row], production=False)["rejected"] == 1
