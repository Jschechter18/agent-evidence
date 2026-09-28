from collections import Counter
from unittest.mock import Mock

import pytest

from mas_sae.data import musique


def make_rows(counts: dict[str, int]) -> list[dict]:
    """Build a fake MuSiQue split with the given number of ids per hop type."""
    rows = []

    for hop, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "id": f"{hop}__{index}_{index + 1}",
                    "question": f"{hop} question {index}",
                    "paragraphs": [],
                    "answer": str(index),
                    "answer_aliases": [],
                    "answerable": True,
                }
            )

    return rows


POOL = make_rows({"2hop": 40, "3hop1": 12, "3hop2": 8, "4hop1": 5, "4hop2": 3})
HOP_PROPORTIONS = {"2hop": 0.6, "3hop": 0.3, "4hop": 0.1}
SPLIT_PROPORTIONS = {
    "discovery": 0.6,
    "validation": 0.2,
    "intervention": 0.2,
}


@pytest.fixture
def fake_dataset(monkeypatch):
    mock_load_dataset = Mock(return_value=POOL)
    monkeypatch.setattr(musique, "load_dataset", mock_load_dataset)
    return mock_load_dataset


def test_hop_type_and_group() -> None:
    assert musique.hop_type("2hop__482757_12019") == "2hop"
    assert musique.hop_type("3hop1__1_2") == "3hop1"
    assert musique.hop_group("3hop1__1_2") == "3hop"
    assert musique.hop_group("3hop2__1_2") == "3hop"
    assert musique.hop_group("4hop3__1_2") == "4hop"
    assert musique.hop_group("weird-id") == "unknown"


def test_allocate_counts_is_exact_and_deterministic() -> None:
    counts = musique.allocate_counts(10, HOP_PROPORTIONS)

    assert counts == {"2hop": 6, "3hop": 3, "4hop": 1}
    assert sum(counts.values()) == 10

    # 3 questions: 1.8 / 0.9 / 0.3 -> largest remainders go to 3hop then 4hop
    assert musique.allocate_counts(3, HOP_PROPORTIONS) == {
        "2hop": 2,
        "3hop": 1,
        "4hop": 0,
    }
    assert musique.allocate_counts(0, HOP_PROPORTIONS) == {
        "2hop": 0,
        "3hop": 0,
        "4hop": 0,
    }


@pytest.mark.parametrize(
    ("proportions", "message"),
    [
        ({}, "non-empty mapping"),
        ({"2hop": 0.5, "3hop": 0.4}, "must sum to 1.0"),
        ({"2hop": 1.5, "3hop": -0.5}, "greater than zero"),
        ({"2hop": True}, "must be a number"),
    ],
)
def test_validate_proportions_rejects_invalid(
    proportions: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        musique.validate_proportions(proportions, name="proportions")


def test_stratified_sample_matches_hop_proportions(fake_dataset) -> None:
    examples = musique.sample_musique_examples(
        source_split="train",
        num_questions=20,
        seed=7,
        hop_proportions=HOP_PROPORTIONS,
    )

    fake_dataset.assert_called_once_with(
        musique.MUSIQUE_DATASET_ID, split="train"
    )

    ids = [example["id"] for example in examples]
    groups = Counter(musique.hop_group(question_id) for question_id in ids)

    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert groups == {"2hop": 12, "3hop": 6, "4hop": 2}


def test_stratified_sample_same_seed_same_ids(fake_dataset) -> None:
    first = musique.sample_musique_examples(
        source_split="train",
        num_questions=15,
        seed=42,
        hop_proportions=HOP_PROPORTIONS,
    )
    second = musique.sample_musique_examples(
        source_split="train",
        num_questions=15,
        seed=42,
        hop_proportions=HOP_PROPORTIONS,
    )

    assert [e["id"] for e in first] == [e["id"] for e in second]


def test_stratified_sample_different_seed_can_differ(fake_dataset) -> None:
    first = musique.sample_musique_examples(
        source_split="train",
        num_questions=15,
        seed=1,
        hop_proportions=HOP_PROPORTIONS,
    )
    second = musique.sample_musique_examples(
        source_split="train",
        num_questions=15,
        seed=2,
        hop_proportions=HOP_PROPORTIONS,
    )

    assert [e["id"] for e in first] != [e["id"] for e in second]


def test_stratified_sample_is_not_first_n(fake_dataset) -> None:
    examples = musique.sample_musique_examples(
        source_split="train",
        num_questions=10,
        seed=3,
        hop_proportions=HOP_PROPORTIONS,
    )
    first_n_ids = [row["id"] for row in POOL[:10]]

    assert [e["id"] for e in examples] != first_n_ids
    assert {musique.hop_group(e["id"]) for e in examples} == {
        "2hop",
        "3hop",
        "4hop",
    }


def test_stratified_sample_rejects_insufficient_stratum(fake_dataset) -> None:
    # pool has 8 4hop questions; 0.9 * 10 = 9 requested
    with pytest.raises(RuntimeError, match="Requested 9 4hop questions"):
        musique.sample_musique_examples(
            source_split="train",
            num_questions=10,
            seed=0,
            hop_proportions={"2hop": 0.1, "4hop": 0.9},
        )


def test_random_sample_without_strata_is_seeded(fake_dataset) -> None:
    first = musique.sample_musique_examples(
        source_split="validation",
        num_questions=9,
        seed=11,
    )
    second = musique.sample_musique_examples(
        source_split="validation",
        num_questions=9,
        seed=11,
    )

    assert [e["id"] for e in first] == [e["id"] for e in second]
    assert len({e["id"] for e in first}) == 9


def test_random_sample_rejects_too_many_questions(fake_dataset) -> None:
    with pytest.raises(RuntimeError, match="only found 68"):
        musique.sample_musique_examples(
            source_split="train",
            num_questions=69,
            seed=0,
        )


def test_sample_skips_unanswerable(monkeypatch) -> None:
    rows = make_rows({"2hop": 4})
    rows[0]["answerable"] = False
    monkeypatch.setattr(musique, "load_dataset", Mock(return_value=rows))

    examples = musique.sample_musique_examples(
        source_split="train",
        num_questions=3,
        seed=0,
    )

    assert rows[0]["id"] not in {e["id"] for e in examples}


def test_sample_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="Unsupported MuSiQue source split"):
        musique.sample_musique_examples(
            source_split="test", num_questions=1, seed=0
        )

    with pytest.raises(ValueError, match="greater than zero"):
        musique.sample_musique_examples(
            source_split="train", num_questions=0, seed=0
        )


def test_assign_experiment_splits_is_question_level_and_deterministic() -> None:
    ids = [row["id"] for row in POOL[:30]]

    first = musique.assign_experiment_splits(
        ids, proportions=SPLIT_PROPORTIONS, seed=5
    )
    second = musique.assign_experiment_splits(
        ids, proportions=SPLIT_PROPORTIONS, seed=5
    )

    assert first == second
    assert set(first) == set(ids)
    assert set(first.values()) <= set(musique.SUPPORTED_EXPERIMENT_SPLITS)

    counts = Counter(first.values())
    assert counts == {"discovery": 18, "validation": 6, "intervention": 6}


def test_assign_experiment_splits_keeps_hop_mix_per_split() -> None:
    ids = [row["id"] for row in make_rows({"2hop": 20, "3hop1": 10})]

    assignment = musique.assign_experiment_splits(
        ids, proportions=SPLIT_PROPORTIONS, seed=9
    )

    by_split_and_group = Counter(
        (split, musique.hop_group(question_id))
        for question_id, split in assignment.items()
    )

    assert by_split_and_group[("discovery", "2hop")] == 12
    assert by_split_and_group[("discovery", "3hop")] == 6
    assert by_split_and_group[("intervention", "2hop")] == 4
    assert by_split_and_group[("intervention", "3hop")] == 2


def test_assign_experiment_splits_different_seed_can_differ() -> None:
    ids = [row["id"] for row in POOL[:30]]

    first = musique.assign_experiment_splits(
        ids, proportions=SPLIT_PROPORTIONS, seed=1
    )
    second = musique.assign_experiment_splits(
        ids, proportions=SPLIT_PROPORTIONS, seed=2
    )

    assert first != second


def test_assign_experiment_splits_rejects_duplicates_and_bad_names() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        musique.assign_experiment_splits(
            ["2hop__1_2", "2hop__1_2"],
            proportions=SPLIT_PROPORTIONS,
            seed=0,
        )

    with pytest.raises(ValueError, match="unsupported names"):
        musique.assign_experiment_splits(
            ["2hop__1_2"],
            proportions={"train": 0.5, "test": 0.5},
            seed=0,
        )


def test_describe_sampled_questions_records_order_and_split() -> None:
    examples = [
        {"id": "3hop1__1_2"},
        {"id": "2hop__3_4"},
    ]
    splits = {"3hop1__1_2": "discovery", "2hop__3_4": "intervention"}

    manifest = musique.describe_sampled_questions(examples, splits)

    assert manifest == [
        {
            "position": 0,
            "question_id": "3hop1__1_2",
            "hop_type": "3hop1",
            "hop_group": "3hop",
            "experiment_split": "discovery",
        },
        {
            "position": 1,
            "question_id": "2hop__3_4",
            "hop_type": "2hop",
            "hop_group": "2hop",
            "experiment_split": "intervention",
        },
    ]

    without_split = musique.describe_sampled_questions(examples)
    assert "experiment_split" not in without_split[0]


def test_load_musique_examples_first_n_unchanged(fake_dataset) -> None:
    """V1 behaviour: first N answerable rows in dataset order."""
    examples = musique.load_musique_examples(
        source_split="train", num_questions=5
    )

    assert [e["id"] for e in examples] == [row["id"] for row in POOL[:5]]
