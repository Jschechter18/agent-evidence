from __future__ import annotations

from pathlib import Path
from typing import Any, NotRequired, TypedDict

import yaml

from mas_sae.data.musique import (
    SUPPORTED_SOURCE_SPLITS,
    validate_experiment_split_proportions,
    validate_proportions,
)


SUPPORTED_SAMPLING_STRATEGIES = ("first_n", "random", "stratified")


class ModelConfig(TypedDict):
    id: str


class SamplingConfig(TypedDict):
    """How questions are drawn from the MuSiQue source split.

    ``first_n`` is the original behaviour (dataset order). ``random`` draws
    a seeded uniform sample. ``stratified`` draws a seeded sample with the
    given ``hop_proportions`` over hop groups (``2hop``, ``3hop``, ``4hop``).
    ``seed`` defaults to ``collection.seed`` when omitted.
    """

    strategy: str
    seed: NotRequired[int]
    hop_proportions: NotRequired[dict[str, float]]


class ExperimentSplitConfig(TypedDict):
    """Question-level scientific split, separate from ``source_split``.

    ``proportions`` maps a subset of ``discovery`` / ``validation`` /
    ``intervention`` to fractions summing to one. ``seed`` defaults to
    ``collection.seed`` when omitted.
    """

    proportions: dict[str, float]
    seed: NotRequired[int]


class DatasetConfig(TypedDict):
    source_split: str
    num_questions: int
    sampling: NotRequired[SamplingConfig]
    experiment_split: NotRequired[ExperimentSplitConfig]


class CollectionSettings(TypedDict):
    """Activation layers, seed, and an optional protocol label.

    ``protocol_version`` is a run label that the collection script maps
    to critic and target behaviour; the loader only checks that it is a
    non-empty string. Omitted means the script's default, so existing
    configs are unchanged.
    """

    layers: list[int]
    seed: int
    protocol_version: NotRequired[str]


class OutputConfig(TypedDict):
    run_name: str


class CollectionConfig(TypedDict):
    model: ModelConfig
    dataset: DatasetConfig
    collection: CollectionSettings
    output: OutputConfig


def load_collection_config(path: str | Path) -> CollectionConfig:
    """Load and validate one activation-collection YAML config."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Collection config must be a mapping.")

    required_sections = {"model", "dataset", "collection", "output"}
    missing = required_sections - raw.keys()

    if missing:
        raise ValueError(
            f"Collection config is missing sections: {sorted(missing)}."
        )

    for section in required_sections:
        if not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a mapping.")

    model = raw["model"]
    dataset = raw["dataset"]
    collection = raw["collection"]
    output = raw["output"]

    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model.id must be a non-empty string.")

    source_split = dataset.get("source_split")
    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            "dataset.source_split must be one of "
            f"{sorted(SUPPORTED_SOURCE_SPLITS)}."
        )

    num_questions = dataset.get("num_questions")
    if type(num_questions) is not int or num_questions <= 0:
        raise ValueError(
            "dataset.num_questions must be a positive integer."
        )

    if "sampling" in dataset:
        _validate_sampling(dataset["sampling"])

    if "experiment_split" in dataset:
        _validate_experiment_split(dataset["experiment_split"])

    layers = collection.get("layers")
    if (
        not isinstance(layers, list)
        or not layers
        or not all(type(layer) is int and layer >= 0 for layer in layers)
    ):
        raise ValueError(
            "collection.layers must be a non-empty list "
            "of non-negative integers."
        )

    if len(set(layers)) != len(layers):
        raise ValueError("collection.layers must not contain duplicates.")

    seed = collection.get("seed")
    if type(seed) is not int:
        raise ValueError("collection.seed must be an integer.")

    if "protocol_version" in collection:
        protocol_version = collection["protocol_version"]
        if (
            not isinstance(protocol_version, str)
            or not protocol_version.strip()
        ):
            raise ValueError(
                "collection.protocol_version must be a non-empty string "
                "when provided."
            )

    run_name = output.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("output.run_name must be a non-empty string.")

    return raw


def _validate_optional_seed(section: dict[str, Any], name: str) -> None:
    if "seed" in section and type(section["seed"]) is not int:
        raise ValueError(f"{name}.seed must be an integer when provided.")


def _validate_sampling(sampling: Any) -> None:
    """Validate the optional ``dataset.sampling`` section."""
    if not isinstance(sampling, dict):
        raise ValueError("dataset.sampling must be a mapping.")

    strategy = sampling.get("strategy")
    if strategy not in SUPPORTED_SAMPLING_STRATEGIES:
        raise ValueError(
            "dataset.sampling.strategy must be one of "
            f"{list(SUPPORTED_SAMPLING_STRATEGIES)}."
        )

    _validate_optional_seed(sampling, "dataset.sampling")

    has_proportions = "hop_proportions" in sampling

    if strategy == "stratified" and not has_proportions:
        raise ValueError(
            "dataset.sampling.hop_proportions is required when "
            "strategy is 'stratified'."
        )

    if strategy != "stratified" and has_proportions:
        raise ValueError(
            "dataset.sampling.hop_proportions is only allowed when "
            "strategy is 'stratified'."
        )

    if has_proportions:
        proportions = sampling["hop_proportions"]
        validate_proportions(
            proportions, name="dataset.sampling.hop_proportions"
        )

        for group in proportions:
            if not group.endswith("hop") or not group[:-3].isdigit():
                raise ValueError(
                    "dataset.sampling.hop_proportions keys must be hop "
                    f"groups such as '2hop', got {group!r}."
                )


def _validate_experiment_split(experiment_split: Any) -> None:
    """Validate the optional ``dataset.experiment_split`` section."""
    if not isinstance(experiment_split, dict):
        raise ValueError("dataset.experiment_split must be a mapping.")

    if "proportions" not in experiment_split:
        raise ValueError(
            "dataset.experiment_split.proportions is required."
        )

    validate_experiment_split_proportions(
        experiment_split["proportions"],
        name="dataset.experiment_split.proportions",
    )

    _validate_optional_seed(experiment_split, "dataset.experiment_split")
