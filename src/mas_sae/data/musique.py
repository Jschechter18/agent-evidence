# src/mas_sae/data/musique.py

import math
import random
import re
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

from datasets import load_dataset


MUSIQUE_DATASET_ID = "dgslibisey/MuSiQue"
SUPPORTED_SOURCE_SPLITS = {"train", "validation"}
SUPPORTED_EXPERIMENT_SPLITS = ("discovery", "validation", "intervention")
PROPORTION_TOLERANCE = 1e-6

_HOP_GROUP_PATTERN = re.compile(r"^(\d+)hop")


class MuSiQueExample(TypedDict):
    """MuSiQue fields required by activation collection."""

    id: str
    question: str
    paragraphs: list[dict[str, Any]]
    answer: str
    answer_aliases: NotRequired[list[str]]
    answerable: NotRequired[bool]


def download_musique(output_dir: Path) -> tuple[Path, Path]:
    """Helper function to download MuSiQue dataset. This dataset is currently a community maintained one from huggingface. We will use this for the smoke test, but load
    from the actual source after early development.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory where the MuSiQue dataset will be saved

    Returns
    -------
    tuple[Path, Path]
        Paths to the train and validation JSON files.
    """
    dataset = load_dataset(MUSIQUE_DATASET_ID)

    train = dataset["train"]
    validation = dataset["validation"]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.json"
    validation_file = output_dir / "validation.json"


    train.to_json(train_file)
    validation.to_json(validation_file)

    return train_file, validation_file


def load_musique_examples(
    source_split: str,
    num_questions: int | None = None,
) -> list[MuSiQueExample]:
    """Load answerable MuSiQue examples from one source split in dataset order."""
    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            f"Unsupported MuSiQue source split: {source_split!r}. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_SPLITS)}."
        )

    if num_questions is not None and num_questions <= 0:
        raise ValueError("num_questions must be greater than zero.")

    dataset = load_dataset(MUSIQUE_DATASET_ID, split=source_split)
    examples: list[MuSiQueExample] = []

    for example in dataset:
        if not example.get("answerable", True):
            continue

        examples.append(cast(MuSiQueExample, dict(example)))

        if num_questions is not None and len(examples) >= num_questions:
            break

    if num_questions is not None and len(examples) < num_questions:
        raise RuntimeError(
            f"Requested {num_questions} answerable questions from "
            f"{source_split!r}, but only found {len(examples)}."
        )

    return examples


def hop_type(question_id: str) -> str:
    """Return the fine-grained MuSiQue hop type encoded in a question id.

    MuSiQue ids look like ``2hop__482757_12019`` or ``3hop1__...``. The
    prefix before the double underscore is the hop type.
    """
    return str(question_id).split("__", 1)[0]


def hop_group(question_id: str) -> str:
    """Return the coarse hop group (``2hop``, ``3hop``, ``4hop``) of a question id.

    Fine-grained types such as ``3hop1`` and ``3hop2`` map to ``3hop``.
    Ids without a recognisable prefix map to ``unknown``.
    """
    match = _HOP_GROUP_PATTERN.match(hop_type(question_id))

    if match is None:
        return "unknown"

    return f"{int(match.group(1))}hop"


def validate_proportions(
    proportions: dict[str, float],
    *,
    name: str,
) -> None:
    """Validate that proportions are positive numbers summing to one."""
    if not isinstance(proportions, dict) or not proportions:
        raise ValueError(f"{name} must be a non-empty mapping.")

    for key, value in proportions.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{key!r}] must be a number.")
        if value <= 0:
            raise ValueError(f"{name}[{key!r}] must be greater than zero.")

    total = float(sum(proportions.values()))

    if abs(total - 1.0) > PROPORTION_TOLERANCE:
        raise ValueError(f"{name} must sum to 1.0, got {total}.")


def validate_experiment_split_proportions(
    proportions: dict[str, float],
    *,
    name: str = "experiment_split.proportions",
) -> None:
    """Validate split proportions and that names are supported splits."""
    validate_proportions(proportions, name=name)

    unknown = set(proportions) - set(SUPPORTED_EXPERIMENT_SPLITS)
    if unknown:
        raise ValueError(
            f"{name} has unsupported names {sorted(unknown)}; expected a "
            f"subset of {list(SUPPORTED_EXPERIMENT_SPLITS)}."
        )


def allocate_counts(
    total: int,
    proportions: dict[str, float],
) -> dict[str, int]:
    """Split ``total`` into integer counts by largest-remainder rounding.

    Ties in the fractional remainder are broken by mapping order so the
    allocation is deterministic for a given config.
    """
    if total < 0:
        raise ValueError("total must be non-negative.")

    validate_proportions(proportions, name="proportions")

    raw = {key: total * float(value) for key, value in proportions.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    keys = list(proportions)

    ranked = sorted(
        keys,
        key=lambda key: (-(raw[key] - counts[key]), keys.index(key)),
    )

    for key in ranked[:remaining]:
        counts[key] += 1

    return counts


def sample_musique_examples(
    *,
    source_split: str,
    num_questions: int,
    seed: int,
    hop_proportions: dict[str, float] | None = None,
) -> list[MuSiQueExample]:
    """Draw a seeded random sample of answerable MuSiQue questions.

    Parameters
    ----------
    source_split
        MuSiQue source split to sample from (``train`` or ``validation``).
    num_questions
        Number of unique questions to select.
    seed
        Seed for the sampler. The same seed, split and proportions always
        select the same question ids in the same order.
    hop_proportions
        Optional mapping from hop group (``2hop``, ``3hop``, ``4hop``) to the
        fraction of the sample drawn from that group. When omitted, the sample
        is drawn uniformly from the whole answerable pool.

    Returns
    -------
    list[MuSiQueExample]
        Selected examples in a seeded shuffled order, so a partial run still
        covers a mix of hop groups.

    Raises
    ------
    ValueError
        On an unsupported split, non-positive size, or invalid proportions.
    RuntimeError
        If the pool, or any hop group, has fewer questions than requested.
    """
    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            f"Unsupported MuSiQue source split: {source_split!r}. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_SPLITS)}."
        )

    if num_questions <= 0:
        raise ValueError("num_questions must be greater than zero.")

    if hop_proportions is not None:
        validate_proportions(hop_proportions, name="hop_proportions")

    dataset = load_dataset(MUSIQUE_DATASET_ID, split=source_split)
    pool: list[MuSiQueExample] = [
        cast(MuSiQueExample, dict(example))
        for example in dataset
        if example.get("answerable", True)
    ]

    rng = random.Random(seed)

    if hop_proportions is None:
        if len(pool) < num_questions:
            raise RuntimeError(
                f"Requested {num_questions} answerable questions from "
                f"{source_split!r}, but only found {len(pool)}."
            )
        selected = rng.sample(pool, num_questions)
        return selected

    counts = allocate_counts(num_questions, hop_proportions)
    by_group: dict[str, list[MuSiQueExample]] = {
        group: [] for group in hop_proportions
    }

    for example in pool:
        group = hop_group(example["id"])
        if group in by_group:
            by_group[group].append(example)

    for group, count in counts.items():
        if len(by_group[group]) < count:
            raise RuntimeError(
                f"Requested {count} {group} questions from "
                f"{source_split!r}, but only found {len(by_group[group])}."
            )

    selected = []

    for group in hop_proportions:
        selected.extend(rng.sample(by_group[group], counts[group]))

    rng.shuffle(selected)
    return selected


def assign_experiment_splits(
    question_ids: list[str],
    *,
    proportions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign every question id to exactly one experiment split.

    Assignment is done at question level, never at episode level, so the
    three critic episodes generated from one question always share a split.
    Within each hop group the ids are shuffled with ``seed`` and cut by
    ``proportions`` (largest-remainder rounding), so each split keeps the
    sample's hop mix.

    Parameters
    ----------
    question_ids
        Unique MuSiQue question ids in collection order.
    proportions
        Mapping from experiment split name to fraction. Names must be a
        subset of ``SUPPORTED_EXPERIMENT_SPLITS``.
    seed
        Seed for the shuffle inside each hop group.

    Returns
    -------
    dict[str, str]
        Mapping from question id to experiment split.
    """
    validate_experiment_split_proportions(proportions)

    ids = [str(question_id) for question_id in question_ids]

    if len(set(ids)) != len(ids):
        raise ValueError("question_ids must not contain duplicates.")

    by_group: dict[str, list[str]] = {}
    for question_id in ids:
        by_group.setdefault(hop_group(question_id), []).append(question_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}

    for group in sorted(by_group):
        members = list(by_group[group])
        rng.shuffle(members)
        counts = allocate_counts(len(members), proportions)
        start = 0

        for split_name in proportions:
            end = start + counts[split_name]
            for question_id in members[start:end]:
                assignment[question_id] = split_name
            start = end

    return assignment


def describe_sampled_questions(
    examples: list[MuSiQueExample],
    experiment_splits: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build the reproducibility manifest of sampled questions in order."""
    manifest: list[dict[str, Any]] = []

    for position, example in enumerate(examples):
        question_id = str(example["id"])
        entry: dict[str, Any] = {
            "position": position,
            "question_id": question_id,
            "hop_type": hop_type(question_id),
            "hop_group": hop_group(question_id),
        }

        if experiment_splits is not None:
            entry["experiment_split"] = experiment_splits[question_id]

        manifest.append(entry)

    return manifest
