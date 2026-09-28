import logging
from unittest.mock import Mock

import pytest
import torch

from mas_sae.agents.critic import CriticBlindAnswerError
from mas_sae.experiments import collection
from mas_sae.experiments.controlled_targets import ControlledTargetError


SITES = [
    "model.language_model.layers.8",
    "model.language_model.layers.17",
]


def make_result(question_id: str):
    return {
        "attempt1": "first answer",
        "attempt1_activations": {
            site: torch.ones(1, 4)
            for site in SITES
        },
        "episodes": [
            {
                "record": {
                    "question_id": question_id,
                    "critic_condition": condition,
                    "solver_accepted_feedback": accepted,
                },
                "attempt2_activations": {
                    site: torch.ones(1, 4)
                    for site in SITES
                },
            }
            for condition, accepted in [
                ("natural", True),
                ("controlled_correct", True),
                ("controlled_incorrect", False),
            ]
        ],
    }


def test_collect_examples_preserves_indices_and_source_split(
    monkeypatch,
) -> None:
    mock_run_question = Mock(
        side_effect=[
            make_result("q1"),
            make_result("q2"),
        ]
    )
    monkeypatch.setattr(
        collection,
        "run_question",
        mock_run_question,
    )

    examples = [
        {
            "id": "q1",
            "question": "Question 1",
            "paragraphs": [],
            "answer": "A1",
            "answer_aliases": [],
        },
        {
            "id": "q2",
            "question": "Question 2",
            "paragraphs": [],
            "answer": "A2",
            "answer_aliases": [],
        },
    ]

    result = collection.collect_examples(
        examples=examples,
        source_split="train",
        model=object(),
        solver=Mock(),
        critic=Mock(),
        validator=Mock(),
        candidate_sites=SITES,
        base_seed=42,
    )

    records = result["records"]

    assert len(records) == 6
    assert {
        row["source_split"]
        for row in records
    } == {"train"}

    assert [
        row["attempt1_activation_index"]
        for row in records
    ] == [0, 0, 0, 1, 1, 1]

    assert [
        row["attempt2_activation_index"]
        for row in records
    ] == list(range(6))

    assert [
        call.kwargs["seed"]
        for call in mock_run_question.call_args_list
    ] == [42, 43]


def test_stack_site_activations() -> None:
    activations = {
        site: [
            torch.ones(1, 4),
            torch.zeros(1, 4),
        ]
        for site in SITES
    }

    stacked = collection.stack_site_activations(
        activations
    )

    assert stacked[SITES[0]].shape == (2, 4)
    assert stacked[SITES[1]].shape == (2, 4)


def test_collect_examples_rejects_empty_examples() -> None:
    with pytest.raises(
        ValueError,
        match="examples must not be empty",
    ):
        collection.collect_examples(
            examples=[],
            source_split="validation",
            model=object(),
            solver=Mock(),
            critic=Mock(),
            validator=Mock(),
            candidate_sites=SITES,
            base_seed=42,
        )


def test_collect_examples_writes_question_level_experiment_split(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        collection,
        "run_question",
        Mock(side_effect=[make_result("q1"), make_result("q2")]),
    )

    examples = [
        {"id": "q1", "question": "Q1", "paragraphs": [], "answer": "A"},
        {"id": "q2", "question": "Q2", "paragraphs": [], "answer": "B"},
    ]

    result = collection.collect_examples(
        examples=examples,
        source_split="train",
        model=object(),
        solver=Mock(),
        critic=Mock(),
        validator=Mock(),
        candidate_sites=SITES,
        base_seed=42,
        experiment_splits={"q1": "discovery", "q2": "intervention"},
    )

    records = result["records"]

    # all three critic episodes of a question share its split
    assert [row["experiment_split"] for row in records] == [
        "discovery", "discovery", "discovery",
        "intervention", "intervention", "intervention",
    ]
    assert {row["source_split"] for row in records} == {"train"}

    per_question = {}
    for row in records:
        per_question.setdefault(row["question_id"], set()).add(
            row["experiment_split"]
        )
    assert all(len(splits) == 1 for splits in per_question.values())


def test_collect_examples_without_splits_has_no_field(monkeypatch) -> None:
    monkeypatch.setattr(
        collection, "run_question", Mock(side_effect=[make_result("q1")])
    )

    result = collection.collect_examples(
        examples=[
            {"id": "q1", "question": "Q1", "paragraphs": [], "answer": "A"},
        ],
        source_split="validation",
        model=object(),
        solver=Mock(),
        critic=Mock(),
        validator=Mock(),
        candidate_sites=SITES,
        base_seed=42,
    )

    assert all("experiment_split" not in row for row in result["records"])


def test_collect_examples_rejects_missing_split_assignment() -> None:
    with pytest.raises(ValueError, match="missing question ids"):
        collection.collect_examples(
            examples=[
                {"id": "q1", "question": "Q1", "paragraphs": [], "answer": "A"},
            ],
            source_split="train",
            model=object(),
            solver=Mock(),
            critic=Mock(),
            validator=Mock(),
            candidate_sites=SITES,
            base_seed=42,
            experiment_splits={"other": "discovery"},
        )


def test_select_questions_first_n_is_v1_behaviour(monkeypatch) -> None:
    examples = [{"id": "2hop__1_2"}, {"id": "2hop__3_4"}]
    mock_first_n = Mock(return_value=examples)
    mock_sample = Mock()
    monkeypatch.setattr(collection, "load_musique_examples", mock_first_n)
    monkeypatch.setattr(collection, "sample_musique_examples", mock_sample)

    selection = collection.select_questions(
        {"source_split": "train", "num_questions": 2},
        default_seed=42,
    )

    mock_first_n.assert_called_once_with(
        source_split="train", num_questions=2
    )
    mock_sample.assert_not_called()
    assert selection["examples"] == examples
    assert selection["experiment_splits"] is None
    assert selection["sampled_questions"] is None


def test_select_questions_stratified_with_experiment_split(
    monkeypatch,
) -> None:
    examples = [
        {"id": "2hop__1_2"},
        {"id": "3hop1__3_4"},
        {"id": "2hop__5_6"},
    ]
    mock_sample = Mock(return_value=examples)
    monkeypatch.setattr(collection, "sample_musique_examples", mock_sample)
    monkeypatch.setattr(
        collection, "load_musique_examples", Mock(side_effect=AssertionError)
    )

    dataset_config = {
        "source_split": "train",
        "num_questions": 3,
        "sampling": {
            "strategy": "stratified",
            "hop_proportions": {"2hop": 0.7, "3hop": 0.3},
        },
        "experiment_split": {
            "proportions": {"discovery": 0.5, "intervention": 0.5},
            "seed": 9,
        },
    }

    selection = collection.select_questions(dataset_config, default_seed=42)

    mock_sample.assert_called_once_with(
        source_split="train",
        num_questions=3,
        seed=42,
        hop_proportions={"2hop": 0.7, "3hop": 0.3},
    )

    splits = selection["experiment_splits"]
    assert set(splits) == {"2hop__1_2", "3hop1__3_4", "2hop__5_6"}
    assert set(splits.values()) <= {"discovery", "intervention"}

    manifest = selection["sampled_questions"]
    assert [entry["question_id"] for entry in manifest] == [
        e["id"] for e in examples
    ]
    assert all(
        entry["experiment_split"] == splits[entry["question_id"]]
        for entry in manifest
    )

    # same config -> same assignment
    again = collection.select_questions(dataset_config, default_seed=42)
    assert again["experiment_splits"] == splits


def test_collect_examples_forwards_decomposition_and_target_flag(
    monkeypatch,
) -> None:
    mock_run_question = Mock(side_effect=[make_result("q1"), make_result("q2")])
    monkeypatch.setattr(collection, "run_question", mock_run_question)

    decomposition = [
        {"question": "Green >> performer", "answer": "Steve Hillage"},
        {"question": "#1 >> spouse", "answer": "Miquette Giraudy"},
    ]
    examples = [
        {
            "id": "q1",
            "question": "Q1",
            "paragraphs": [],
            "answer": "A",
            "question_decomposition": decomposition,
        },
        {"id": "q2", "question": "Q2", "paragraphs": [], "answer": "B"},
    ]

    collection.collect_examples(
        examples=examples,
        source_split="train",
        model=object(),
        solver=Mock(),
        critic=Mock(),
        validator=Mock(),
        candidate_sites=SITES,
        base_seed=42,
        type_checked_target=True,
    )

    first, second = mock_run_question.call_args_list
    assert first.kwargs["decomposition"] == decomposition
    assert first.kwargs["type_checked_target"] is True
    assert second.kwargs["decomposition"] == []
    assert second.kwargs["type_checked_target"] is True


@pytest.mark.parametrize(
    "error",
    [
        CriticBlindAnswerError("no usable blind answer"),
        ControlledTargetError("q2", ["type incompatible"]),
    ],
    ids=["blind_answer", "controlled_target"],
)
def test_collect_examples_skips_question_on_expected_treatment_error(
    monkeypatch, caplog, error
) -> None:
    mock_run_question = Mock(
        side_effect=[make_result("q1"), error, make_result("q3")]
    )
    monkeypatch.setattr(collection, "run_question", mock_run_question)

    examples = [
        {"id": "q1", "question": "Q1", "paragraphs": [], "answer": "A"},
        {"id": "q2", "question": "Q2", "paragraphs": [], "answer": "B"},
        {"id": "q3", "question": "Q3", "paragraphs": [], "answer": "C"},
    ]

    with caplog.at_level(
        logging.WARNING, logger="mas_sae.experiments.collection"
    ):
        result = collection.collect_examples(
            examples=examples,
            source_split="train",
            model=object(),
            solver=Mock(),
            critic=Mock(),
            validator=Mock(),
            candidate_sites=SITES,
            base_seed=42,
        )

    # the failure did not stop collection: every question was attempted,
    # each with the seed it would have had anyway
    assert [
        call.kwargs["seed"] for call in mock_run_question.call_args_list
    ] == [42, 43, 44]

    # the failed question contributes no records ...
    records = result["records"]
    assert len(records) == 6
    assert {row["question_id"] for row in records} == {"q1", "q3"}

    # ... and no activation rows, so indices stay aligned with the tensors
    for site in SITES:
        assert len(result["attempt1_by_site"][site]) == 2
        assert len(result["attempt2_by_site"][site]) == 6
    assert [row["attempt1_activation_index"] for row in records] == [
        0, 0, 0, 1, 1, 1,
    ]
    assert [row["attempt2_activation_index"] for row in records] == list(
        range(6)
    )

    warnings = [
        record for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "q2" in warnings[0].getMessage()
    assert str(error) in warnings[0].getMessage()


def test_collect_examples_propagates_unexpected_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        collection,
        "run_question",
        Mock(side_effect=[make_result("q1"), RuntimeError("boom")]),
    )

    with pytest.raises(RuntimeError, match="boom"):
        collection.collect_examples(
            examples=[
                {"id": "q1", "question": "Q1", "paragraphs": [], "answer": "A"},
                {"id": "q2", "question": "Q2", "paragraphs": [], "answer": "B"},
            ],
            source_split="train",
            model=object(),
            solver=Mock(),
            critic=Mock(),
            validator=Mock(),
            candidate_sites=SITES,
            base_seed=42,
        )


def test_collect_examples_defaults_to_heuristic_target(monkeypatch) -> None:
    mock_run_question = Mock(side_effect=[make_result("q1")])
    monkeypatch.setattr(collection, "run_question", mock_run_question)

    collection.collect_examples(
        examples=[
            {"id": "q1", "question": "Q1", "paragraphs": [], "answer": "A"},
        ],
        source_split="train",
        model=object(),
        solver=Mock(),
        critic=Mock(),
        validator=Mock(),
        candidate_sites=SITES,
        base_seed=42,
    )

    assert mock_run_question.call_args.kwargs["type_checked_target"] is False
