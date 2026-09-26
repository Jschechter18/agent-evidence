import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments import artifacts, collection_artifacts
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



def test_build_resolved_config_adds_provenance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "get_git_commit",
        lambda: "abc123",
    )
    monkeypatch.setattr(
        artifacts,
        "get_package_version",
        lambda package: f"{package}-version",
    )

    config = {
        "model": {"id": "google/gemma-3-4b-it"},
        "dataset": {"source_split": "train", "num_questions": 2},
        "collection": {"layers": [8, 17], "seed": 42},
        "output": {"run_name": "test_run"},
    }

    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="model-revision")
    )
    solver = SimpleNamespace(max_new_tokens=32)
    critic = SimpleNamespace(max_new_tokens=128)
    validator = SimpleNamespace(max_new_tokens=4)

    resolved = collection_artifacts.build_resolved_config(
        config,
        model,
        solver,
        critic,
        validator,
    )

    provenance = resolved["provenance"]

    assert resolved["model"] == config["model"]
    assert provenance["git_commit"] == "abc123"
    assert provenance["model_revision"] == "model-revision"
    assert provenance["dataset_id"] == "dgslibisey/MuSiQue"
    assert provenance["package_versions"]["torch"] == "torch-version"
    assert provenance["generation"]["do_sample"] is False
    assert provenance["generation"]["solver_max_new_tokens"] == 32
    assert provenance["generation"]["critic_max_new_tokens"] == 128
    assert provenance["generation"]["validator_max_new_tokens"] == 4


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


def _resolve(monkeypatch, critic, **kwargs):
    monkeypatch.setattr(artifacts, "get_git_commit", lambda: "abc123")
    monkeypatch.setattr(
        artifacts, "get_package_version", lambda package: "x"
    )

    return collection_artifacts.build_resolved_config(
        {"output": {"run_name": "r"}},
        SimpleNamespace(config=SimpleNamespace(_commit_hash="rev")),
        SimpleNamespace(max_new_tokens=32),
        critic,
        SimpleNamespace(max_new_tokens=4),
        **kwargs,
    )["provenance"]


@pytest.mark.parametrize(
    (
        "critic",
        "type_checked_target",
        "expected_natural",
        "expected_controlled",
    ),
    [
        (
            SimpleNamespace(max_new_tokens=128),
            False,
            "NATURAL_PROMPT_V1",
            "CONTROLLED_PROMPT_V1",
        ),
        (
            SimpleNamespace(
                max_new_tokens=128,
                blind_then_compare=False,
                controlled_as_own_conclusion=False,
            ),
            False,
            "NATURAL_PROMPT_V1",
            "CONTROLLED_PROMPT_V1",
        ),
        (
            SimpleNamespace(
                max_new_tokens=128,
                blind_then_compare=True,
                controlled_as_own_conclusion=True,
            ),
            True,
            "NATURAL_BLIND_PROMPT+NATURAL_COMPARE_PROMPT",
            "CONTROLLED_OWN_CONCLUSION_PROMPT",
        ),
        (
            SimpleNamespace(
                max_new_tokens=128,
                blind_then_compare=True,
                controlled_as_own_conclusion=False,
            ),
            False,
            "NATURAL_BLIND_PROMPT+NATURAL_COMPARE_PROMPT",
            "CONTROLLED_PROMPT_V1",
        ),
    ],
    ids=["no-attributes", "all-off", "all-on", "blind-only"],
)
def test_build_resolved_config_records_actual_capabilities(
    monkeypatch,
    critic,
    type_checked_target,
    expected_natural,
    expected_controlled,
) -> None:
    provenance = _resolve(
        monkeypatch, critic, type_checked_target=type_checked_target
    )
    prompts = provenance["prompt_versions"]

    assert provenance["capabilities"] == {
        "blind_then_compare": getattr(critic, "blind_then_compare", False),
        "controlled_as_own_conclusion": getattr(
            critic, "controlled_as_own_conclusion", False
        ),
        "type_checked_target": type_checked_target,
    }
    assert prompts["critic_natural"] == expected_natural
    assert prompts["critic_controlled"] == expected_controlled
    assert prompts["solver_solve"] == "SOLVE_PROMPT_V1"
    assert ("critic_distractor" in prompts) == type_checked_target


def test_build_resolved_config_echoes_protocol_label_without_branching(
    monkeypatch,
) -> None:
    critic = SimpleNamespace(max_new_tokens=128)

    # the label is recorded as given and does not change the behaviour
    # actually recorded
    labelled = _resolve(monkeypatch, critic, protocol_version="v9")
    assert labelled["protocol_version"] == "v9"
    assert labelled["capabilities"]["blind_then_compare"] is False
    assert labelled["prompt_versions"]["critic_natural"] == "NATURAL_PROMPT_V1"

    # no label supplied: no label key, everything else unchanged
    unlabelled = _resolve(monkeypatch, critic)
    assert "protocol_version" not in unlabelled
    assert unlabelled["capabilities"] == labelled["capabilities"]
    assert unlabelled["prompt_versions"] == labelled["prompt_versions"]
