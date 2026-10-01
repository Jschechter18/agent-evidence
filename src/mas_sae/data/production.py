"""Split manifest and safe loading for the production activation data.

Every question belongs to exactly one partition, taken from the full
production run's ``experiment_split``:

    discovery     fitting, choosing layers and features, tuning
    validation    testing a configuration that is already frozen
    intervention  the later causal experiments

The layer scan re-ran 2,500 of the same questions but gave them splits of its
own, and 1,370 of those differ from the full run. The scan's split is recorded
here but never used.

Questions that were part of earlier development data (the layer scan or the
first 100-row QC sample) are flagged. The flag records that a question was
available then, not that it was used in any decision. Held-out results can be
reported with and without these questions.
"""
from collections import Counter
import csv
from pathlib import Path

import torch

from mas_sae.data.activation_store import ActivationStore
from mas_sae.data.musique import SUPPORTED_EXPERIMENT_SPLITS, SUPPORTED_SOURCE_SPLITS
from mas_sae.evaluation.behavior_v01 import LABEL_INPUT_FIELDS, read_labels
from mas_sae.experiments.records import read_jsonl

# What each stated purpose is allowed to read.
PURPOSE_PARTITION = {
    "fit": "discovery",
    "tune": "discovery",
    "evaluate": "validation",
    "intervene": "intervention",
}

MANIFEST_FIELDS = [
    "question_id",
    "source_split",         # MuSiQue split the question came from
    "canonical_split",      # the partition we use
    "scan_split",           # the layer scan's own split; blank if not in the scan
    "in_qc100",             # in the first 100-row QC sample
    "in_development_data",  # in the layer scan or the first QC sample
    "full_index",           # row in the full run's activation files
    "scan_index",           # row in the layer scan's activation files; blank if none
]


def read_run_interactions(results_dir):
    """All episodes of one run: its train and validation ``interactions.jsonl`` together."""
    rows = []
    for source_split in sorted(SUPPORTED_SOURCE_SPLITS):
        rows.extend(read_jsonl(Path(results_dir) / source_split / "interactions.jsonl"))
    return rows


def _activation_index(row):
    """Row of this episode in its run's activation files. A1 and A2 share it."""
    index = row["attempt2_activation_index"]
    if row["attempt1_activation_index"] != index:
        raise ValueError(f"{row['question_id']}: A1 and A2 activation indices differ")
    if type(index) is not int or index < 0:
        raise ValueError(f"{row['question_id']}: activation index must be a non-negative integer")
    return index


def _check_indices_are_unique(rows, run):
    """Two episodes of one run must never point at the same activation row."""
    used = set()
    for row in rows:
        position = (row["source_split"], _activation_index(row))
        if position in used:
            raise ValueError(f"{row['question_id']}: {run} run reuses activation row {position}")
        used.add(position)


def canonical_manifest(full_rows, scan_rows=(), qc100_ids=()):
    """Build {question_id: record} from the full run, the layer scan and the old QC sample."""
    qc100_ids = set(qc100_ids)
    _check_indices_are_unique(full_rows, "full")
    _check_indices_are_unique(scan_rows, "scan")
    manifest = {}

    for row in full_rows:
        qid = row["question_id"]
        if qid in manifest:
            raise ValueError(f"{qid}: the full run must have one episode per question")
        if row["experiment_split"] not in SUPPORTED_EXPERIMENT_SPLITS:
            raise ValueError(f"{qid}: unknown experiment split {row['experiment_split']!r}")
        manifest[qid] = {
            "question_id": qid,
            "source_split": row["source_split"],
            "canonical_split": row["experiment_split"],
            "scan_split": "",
            "in_qc100": qid in qc100_ids,
            "in_development_data": qid in qc100_ids,
            "full_index": _activation_index(row),
            "scan_index": None,
        }

    for row in scan_rows:
        qid = row["question_id"]
        record = manifest.get(qid)
        if record is None or record["scan_split"]:
            raise ValueError(f"{qid}: scan question missing from the full run or repeated")
        if record["source_split"] != row["source_split"]:
            raise ValueError(f"{qid}: question crosses source splits")
        record["scan_split"] = row["experiment_split"]
        record["scan_index"] = _activation_index(row)
        record["in_development_data"] = True

    return manifest


def check_scan_repeats_full_run(full_rows, scan_rows):
    """The scan must repeat the full run on every field that decides a label.

    Otherwise the full run's labels would not apply to the scan's episodes.
    """
    full_by_question = {row["question_id"]: row for row in full_rows}
    for row in scan_rows:
        full = full_by_question[row["question_id"]]
        for field in LABEL_INPUT_FIELDS:
            if row[field] != full[field]:
                raise ValueError(f"{row['question_id']}: scan and full run disagree on {field}")


def split_audit(manifest):
    """How the layer scan and the first QC sample overlap with the full run's split."""
    records = list(manifest.values())
    in_scan = [m for m in records if m["scan_split"]]
    different = [m for m in in_scan if m["scan_split"] != m["canonical_split"]]
    leaking = [m for m in different if m["scan_split"] == "discovery"]
    in_qc100 = [m for m in records if m["in_qc100"]]
    in_development = [m for m in records if m["in_development_data"]]

    def by_split(group):
        return dict(Counter(m["canonical_split"] for m in group))

    return {
        "questions": len(records),
        "by_canonical_split": by_split(records),
        "scan_questions_also_in_full_run": len(in_scan),
        "scan_questions_with_a_different_split": len(different),
        "scan_discovery_but_held_out_in_full_run": len(leaking),
        "qc100_by_canonical_split": by_split(in_qc100),
        "in_development_data_by_canonical_split": by_split(in_development),
    }


def write_manifest(path, manifest):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in manifest.values():
            writer.writerow({k: "" if v is None else v for k, v in record.items()})


def read_manifest(path):
    with open(path, newline="") as f:
        records = list(csv.DictReader(f))
    for record in records:
        for field in ("in_qc100", "in_development_data"):
            record[field] = record[field] == "True"
        for field in ("full_index", "scan_index"):
            record[field] = int(record[field]) if record[field] else None
    return {record["question_id"]: record for record in records}


def authorize_rows(rows, manifest, *, purpose):
    """Raise unless every row is in the partition that ``purpose`` may read.

    The partition is looked up by question ID in the manifest. A row's own
    ``experiment_split`` is ignored, because it is wrong for layer-scan rows.
    """
    if purpose not in PURPOSE_PARTITION:
        raise ValueError(f"purpose must be one of {sorted(PURPOSE_PARTITION)}")
    allowed = PURPOSE_PARTITION[purpose]

    seen = set()
    for row in rows:
        qid = row["question_id"]
        if qid in seen:
            raise ValueError(f"{qid}: repeated question; "
                             "scan and full rows are not independent examples")
        seen.add(qid)
        if manifest[qid]["canonical_split"] != allowed:
            raise ValueError(f"{qid}: purpose {purpose!r} requires {allowed} questions")
    return rows


def load_labeled_rows(results_dir, package_dir):
    """Join a run's interactions with the labels and the split manifest.

    ``results_dir`` holds ``train/`` and ``validation/`` folders, each with an
    ``interactions.jsonl`` (full run or layer scan). ``package_dir`` is the
    folder written by ``scripts/prepare_behavior_v01.py``.

    Returns ``(rows, manifest)``. Each row is the original interaction with two
    additions: ``canonical_split`` and ``label``.
    """
    package_dir = Path(package_dir)
    manifest = read_manifest(package_dir / "split_manifest.csv")
    labels = read_labels(package_dir / "labels.csv")

    rows = read_run_interactions(results_dir)
    for row in rows:
        qid = row["question_id"]
        row["canonical_split"] = manifest[qid]["canonical_split"]
        row["label"] = labels[qid]
    return rows, manifest


def load_production_activations(rows, manifest, activation_root, *,
                                purpose, layer, attempt=2, run="full"):
    """Return one activation vector per row, in row order.

    purpose   'fit', 'tune', 'evaluate' or 'intervene'. Rows from a partition
              that the purpose may not read are refused.
    attempt   1 is before the Critic's feedback, 2 is after.
    run       'full' or 'scan': the run that both ``rows`` and
              ``activation_root`` come from.

    Mixing rows from one run with activation files from the other would
    return another question's activations without any error, so the file size
    and every row's source split and index are checked against the manifest.
    """
    if attempt not in (1, 2):
        raise ValueError("attempt must be 1 or 2")
    if run not in ("full", "scan"):
        raise ValueError("run must be 'full' or 'scan'")
    if not rows:
        raise ValueError("No rows selected")
    authorize_rows(rows, manifest, purpose=purpose)

    store = ActivationStore(Path(activation_root) / f"layer_{int(layer):02d}")
    index_field = f"{run}_index"
    tensors = {}
    vectors = []

    for row in rows:
        qid = row["question_id"]
        split = row["source_split"]

        if split != manifest[qid]["source_split"]:
            raise ValueError(f"{qid}: row's source split does not match the manifest")

        if split not in tensors:
            tensor = store.load_activations(f"{split}_attempt{attempt}")
            questions_in_run = sum(
                record["source_split"] == split and record[index_field] is not None
                for record in manifest.values())
            if tensor.shape[0] != questions_in_run:
                raise ValueError(
                    f"{split} activation file has {tensor.shape[0]} rows but the {run} run "
                    f"has {questions_in_run} {split} questions; "
                    "activation_root is from a different run")
            tensors[split] = tensor

        index = row[f"attempt{attempt}_activation_index"]
        if index != manifest[qid][index_field]:
            raise ValueError(f"{qid}: row's activation index does not match the {run} run")
        vectors.append(tensors[split][index])

    result = torch.stack(vectors)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite activations")
    return result
