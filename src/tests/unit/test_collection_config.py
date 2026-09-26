from pathlib import Path

import pytest

from mas_sae.experiments.collection_config import (
    load_collection_config,
)


def test_load_collection_config(tmp_path: Path) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: train
  num_questions: 10
collection:
  layers: [8, 17, 25, 33]
  seed: 42
output:
  run_name: train_10q
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_collection_config(path)

    assert config["dataset"]["source_split"] == "train"
    assert config["dataset"]["num_questions"] == 10
    assert config["collection"]["layers"] == [8, 17, 25, 33]
    assert config["collection"]["seed"] == 42


def test_collection_config_rejects_unknown_source_split(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: test
  num_questions: 10
collection:
  layers: [8, 17]
  seed: 42
output:
  run_name: bad_split
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source_split",
    ):
        load_collection_config(path)


def test_collection_config_rejects_non_mapping_section(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
dataset:
  source_split: train
  num_questions: 10
collection:
  layers: [8, 17]
  seed: 42
output:
  run_name: bad_model
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model must be a mapping"):
        load_collection_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_questions", "true", "num_questions"),
        ("seed", "true", "seed"),
    ],
)
def test_collection_config_rejects_boolean_integers(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    num_questions = value if field == "num_questions" else "10"
    seed = value if field == "seed" else "42"

    path = tmp_path / "collection.yaml"
    path.write_text(
        f"""
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: train
  num_questions: {num_questions}
collection:
  layers: [8, 17]
  seed: {seed}
output:
  run_name: validation_test
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_collection_config(path)


def test_collection_config_rejects_duplicate_layers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: train
  num_questions: 10
collection:
  layers: [8, 8]
  seed: 42
output:
  run_name: duplicate_layers
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_collection_config(path)


V2_CONFIG = """
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: train
  num_questions: 10
  sampling:
    strategy: stratified
    seed: 7
    hop_proportions:
      2hop: 0.6
      3hop: 0.3
      4hop: 0.1
  experiment_split:
    proportions:
      discovery: 0.6
      validation: 0.2
      intervention: 0.2
collection:
  layers: [8, 17]
  seed: 42
output:
  run_name: v2_test
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "collection.yaml"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_collection_config_accepts_v2_sampling_and_split(
    tmp_path: Path,
) -> None:
    config = load_collection_config(write_config(tmp_path, V2_CONFIG))

    assert config["dataset"]["sampling"]["strategy"] == "stratified"
    assert config["dataset"]["sampling"]["seed"] == 7
    assert config["dataset"]["sampling"]["hop_proportions"]["3hop"] == 0.3
    assert config["dataset"]["experiment_split"]["proportions"] == {
        "discovery": 0.6,
        "validation": 0.2,
        "intervention": 0.2,
    }


def test_collection_config_v1_has_no_sampling_sections(
    tmp_path: Path,
) -> None:
    config = load_collection_config(
        Path("configs/collection/v1/train_100q.yaml")
    )

    assert "sampling" not in config["dataset"]
    assert "experiment_split" not in config["dataset"]


def test_collection_config_v2_smoke_loads() -> None:
    config = load_collection_config(
        Path("configs/collection/v2/smoke_train.yaml")
    )

    assert config["dataset"]["sampling"]["strategy"] == "stratified"
    assert config["dataset"]["experiment_split"]["proportions"]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("strategy: stratified", "strategy: first_last"),
        ("      4hop: 0.1", "      4hop: 0.2"),
        ("      intervention: 0.2", "      test: 0.2"),
        ("    seed: 7", "    seed: true"),
    ],
    ids=["bad-strategy", "hop-sum", "bad-split-name", "bool-seed"],
)
def test_collection_config_rejects_invalid_v2_sections(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    text = V2_CONFIG.replace(replacement, message)

    with pytest.raises(ValueError):
        load_collection_config(write_config(tmp_path, text))


def test_collection_config_stratified_requires_hop_proportions(
    tmp_path: Path,
) -> None:
    text = V2_CONFIG.replace(
        "    hop_proportions:\n      2hop: 0.6\n      3hop: 0.3\n      4hop: 0.1\n",
        "",
    )

    with pytest.raises(ValueError, match="hop_proportions is required"):
        load_collection_config(write_config(tmp_path, text))


def test_collection_config_random_rejects_hop_proportions(
    tmp_path: Path,
) -> None:
    text = V2_CONFIG.replace("strategy: stratified", "strategy: random")

    with pytest.raises(ValueError, match="only allowed when"):
        load_collection_config(write_config(tmp_path, text))


def test_collection_config_protocol_version_defaults_to_absent(
    tmp_path: Path,
) -> None:
    config = load_collection_config(write_config(tmp_path, V2_CONFIG))

    assert "protocol_version" not in config["collection"]


@pytest.mark.parametrize("value", ["v2", "v3", "my_label"])
def test_collection_config_accepts_any_protocol_version_label(
    tmp_path: Path, value: str
) -> None:
    text = V2_CONFIG.replace(
        "  seed: 42\n", f"  seed: 42\n  protocol_version: {value}\n"
    )
    config = load_collection_config(write_config(tmp_path, text))

    # the loader keeps the label as is; the collection script owns the
    # accepted set
    assert config["collection"]["protocol_version"] == value


@pytest.mark.parametrize("value", ["2", "true", '""', "'  '"])
def test_collection_config_rejects_non_string_protocol_version(
    tmp_path: Path, value: str
) -> None:
    text = V2_CONFIG.replace(
        "  seed: 42\n", f"  seed: 42\n  protocol_version: {value}\n"
    )

    with pytest.raises(ValueError, match="protocol_version"):
        load_collection_config(write_config(tmp_path, text))


def test_collection_config_v2_smoke_carries_v2_label() -> None:
    config = load_collection_config(
        Path("configs/collection/v2/smoke_train.yaml")
    )

    assert config["collection"]["protocol_version"] == "v2"


def test_collection_config_v1_has_no_protocol_version() -> None:
    config = load_collection_config(
        Path("configs/collection/v1/train_100q.yaml")
    )

    assert "protocol_version" not in config["collection"]
