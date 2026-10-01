from pathlib import Path
from unittest.mock import Mock

import pytest

from mas_sae.data import musique


def test_download_musique_writes_expected_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_split = Mock()
    validation_split = Mock()

    mock_load_dataset = Mock(
        return_value={
            "train": train_split,
            "validation": validation_split,
        }
    )
    monkeypatch.setattr(
        musique,
        "load_dataset",
        mock_load_dataset,
    )

    output_dir = tmp_path / "MuSiQue" / "clean"

    train_path, validation_path = musique.download_musique(
        output_dir
    )

    expected_train_path = output_dir / "train.json"
    expected_validation_path = output_dir / "validation.json"

    mock_load_dataset.assert_called_once_with(
        musique.MUSIQUE_DATASET_ID
    )
    train_split.to_json.assert_called_once_with(
        expected_train_path
    )
    validation_split.to_json.assert_called_once_with(
        expected_validation_path
    )

    assert output_dir.is_dir()
    assert train_path == expected_train_path
    assert validation_path == expected_validation_path


def test_load_musique_examples_uses_requested_source_split(
    monkeypatch,
) -> None:
    rows = [
        {
            "id": "q0",
            "answerable": False,
        },
        {
            "id": "q1",
            "answerable": True,
        },
        {
            "id": "q2",
            "answerable": True,
        },
        {
            "id": "q3",
            "answerable": True,
        },
    ]

    mock_load_dataset = Mock(
        return_value=rows
    )
    monkeypatch.setattr(
        musique,
        "load_dataset",
        mock_load_dataset,
    )

    examples = musique.load_musique_examples(
        source_split="train",
        num_questions=2,
    )

    mock_load_dataset.assert_called_once_with(
        musique.MUSIQUE_DATASET_ID,
        split="train",
    )

    assert [
        example["id"]
        for example in examples
    ] == ["q1", "q2"]


def test_load_musique_examples_rejects_unknown_split() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported MuSiQue source split",
    ):
        musique.load_musique_examples(
            source_split="test",
            num_questions=1,
        )


def test_load_musique_examples_requires_available_questions(
    monkeypatch,
) -> None:
    mock_load_dataset = Mock(
        return_value=[
            {
                "id": "q1",
                "answerable": True,
            },
        ]
    )
    monkeypatch.setattr(
        musique,
        "load_dataset",
        mock_load_dataset,
    )

    with pytest.raises(
        RuntimeError,
        match="Requested 2 answerable questions",
    ):
        musique.load_musique_examples(
            source_split="validation",
            num_questions=2,
        )


def test_load_musique_examples_by_id_keeps_order_and_pins_revision(
    monkeypatch,
) -> None:
    class CachedRows(list):
        cache_files = [{"filename": "/cache/rev/train.arrow"}]

    rows = CachedRows([
        {"id": "a", "answerable": True},
        {"id": "b", "answerable": True},
        {"id": "c", "answerable": False},
    ])
    mock_load_dataset = Mock(return_value=rows)
    monkeypatch.setattr(musique, "load_dataset", mock_load_dataset)

    examples = musique.load_musique_examples_by_id("train", ["b", "a"], revision="rev")

    assert [example["id"] for example in examples] == ["b", "a"]
    mock_load_dataset.assert_called_once_with(
        musique.MUSIQUE_DATASET_ID, split="train", revision="rev"
    )

    with pytest.raises(ValueError, match="duplicates"):
        musique.load_musique_examples_by_id("train", ["a", "a"])
    with pytest.raises(RuntimeError, match="not found"):
        musique.load_musique_examples_by_id("train", ["a", "zzz"])
    with pytest.raises(RuntimeError, match="not answerable"):
        musique.load_musique_examples_by_id("train", ["c"])
    with pytest.raises(ValueError, match="revision"):
        musique.load_musique_examples_by_id("train", ["a"], revision=" ")



def test_pinned_load_rejects_cached_revision_mismatch(
    monkeypatch,
) -> None:
    requested = "0" * 40
    actual = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"

    class CachedRows(list):
        cache_files = [
            {
                "filename": (
                    f"/cache/dgslibisey___mu_si_que/default/0.0.0/"
                    f"{actual}/mu_si_que-train.arrow"
                )
            }
        ]

    monkeypatch.setattr(
        musique,
        "load_dataset",
        Mock(return_value=CachedRows()),
    )

    with pytest.raises(
        RuntimeError,
        match="does not match pinned revision",
    ):
        musique.load_musique_split(
            "train",
            revision=requested,
        )


def test_unpinned_load_keeps_historical_call(
    monkeypatch,
) -> None:
    mock_load_dataset = Mock(return_value=[])
    monkeypatch.setattr(musique, "load_dataset", mock_load_dataset)

    musique.load_musique_split("validation")

    mock_load_dataset.assert_called_once_with(
        musique.MUSIQUE_DATASET_ID, split="validation"
    )

def test_load_musique_examples_forwards_revision(
    monkeypatch,
) -> None:
    revision = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
    mock_load_split = Mock(
        return_value=[{"id": "q1", "answerable": True}]
    )
    monkeypatch.setattr(
        musique,
        "load_musique_split",
        mock_load_split,
    )

    examples = musique.load_musique_examples(
        "train",
        num_questions=1,
        revision=revision,
    )

    assert [example["id"] for example in examples] == ["q1"]
    mock_load_split.assert_called_once_with(
        "train",
        revision=revision,
    )


def test_sample_musique_examples_forwards_revision(
    monkeypatch,
) -> None:
    revision = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
    mock_load_split = Mock(
        return_value=[
            {"id": "2hop__1_2", "answerable": True}
        ]
    )
    monkeypatch.setattr(
        musique,
        "load_musique_split",
        mock_load_split,
    )

    examples = musique.sample_musique_examples(
        source_split="train",
        num_questions=1,
        seed=42,
        revision=revision,
    )

    assert [example["id"] for example in examples] == [
        "2hop__1_2"
    ]
    mock_load_split.assert_called_once_with(
        "train",
        revision=revision,
    )
