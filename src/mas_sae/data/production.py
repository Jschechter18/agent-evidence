"""Partition manifest and safe loading for the production activation data.

Every question belongs to exactly one partition, decided from its id and its
MuSiQue source split alone:

    train         fitting: the SAE, the probes, the per-layer probes
    validation    tuning: early stopping, layer choice, feature choice
    test          the frozen SAE and probes, scored once
    intervention  the causal experiment, run live, much later

All MuSiQue-validation questions are ``validation``. MuSiQue-train questions
are sorted by id, grouped by hop count and cut with a seeded shuffle into
``train`` / ``test`` / ``intervention``, so the assignment depends only on
the question set, the proportions and the seed, never on the order a run
collected them in. The proportions and seed are configuration, passed in by
the script that builds the package and recorded next to the manifest.

The split a run assigned while collecting (``experiment_split``) and the
layer scan's own split are kept as history and never read for decisions.
Questions that were part of earlier development data (the layer scan or the
first 100-row QC sample) are flagged. The flag records that a question was
available then, not that it was used in any decision. Held-out results can
be reported with and without these questions.
"""
from collections import Counter
import csv
import json
from pathlib import Path

import torch

from mas_sae.data.activation_store import ActivationStore
from mas_sae.data.musique import (
    SUPPORTED_EXPERIMENT_SPLITS,
    SUPPORTED_PARTITIONS,
    SUPPORTED_SOURCE_SPLITS,
    assign_experiment_splits,
    hop_group,
)
from mas_sae.evaluation.behavior_v01 import LABEL_INPUT_FIELDS, read_labels
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.collection_artifacts import _add_sae_indices, _save_and_verify
from mas_sae.experiments.records import read_jsonl, write_jsonl

PARTITIONS = SUPPORTED_PARTITIONS

# MuSiQue-train questions are divided among these; MuSiQue-validation is all "validation".
TRAIN_SOURCE_PARTITIONS = ("train", "test", "intervention")

# What each stated purpose is allowed to read.
PURPOSE_PARTITION = {
    "fit": "train",
    "tune": "validation",
    "evaluate": "test",
    "intervene": "intervention",
}
PARTITION_PURPOSE = {partition: purpose for purpose, partition in PURPOSE_PARTITION.items()}

MANIFEST_FIELDS = [
    "question_id",
    "source_split",         # MuSiQue split the question came from
    "partition",            # the only column anyone reads to choose data
    "hop_group",            # what the stratification used
    "collection_split",     # experiment_split the collection run assigned; history
    "scan_split",           # the layer scan's own split; blank if not in the scan
    "in_qc100",             # in the first 100-row QC sample
    "in_development_data",  # in the layer scan or the first QC sample
    "full_index",           # row in the full run's activation files
    "scan_index",           # row in the layer scan's activation files; blank if none
]

INTERVENTION_IDS_FILE = "intervention_question_ids.json"


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


def assign_partitions(question_ids_by_source, *, proportions, seed):
    """Return {question_id: partition} from the question set, proportions and seed only.

    ``question_ids_by_source`` maps a MuSiQue source split to its question ids.
    ``proportions`` must name exactly ``train``, ``test`` and ``intervention``.
    Ids are sorted before the seeded, hop-stratified cut, so the same
    questions always receive the same partition whatever order they arrive in.
    """
    if set(proportions) != set(TRAIN_SOURCE_PARTITIONS):
        raise ValueError(f"proportions must name exactly {list(TRAIN_SOURCE_PARTITIONS)}, "
                         f"got {sorted(proportions)}")
    # The assigner cuts in mapping order, so fix the order here: equal
    # proportions must give equal assignments however the caller spelt them.
    proportions = {name: proportions[name] for name in TRAIN_SOURCE_PARTITIONS}
    unknown = set(question_ids_by_source) - SUPPORTED_SOURCE_SPLITS
    if unknown:
        raise ValueError(f"unknown source splits {sorted(unknown)}")

    assignment = {str(qid): "validation" for qid in question_ids_by_source.get("validation", ())}
    train_ids = sorted(str(qid) for qid in question_ids_by_source.get("train", ()))
    if set(train_ids) & set(assignment):
        raise ValueError("a question id appears in both source splits")
    assignment.update(assign_experiment_splits(
        train_ids, proportions=proportions, seed=seed,
        supported_splits=TRAIN_SOURCE_PARTITIONS))
    return assignment


def build_partition_manifest(full_rows, scan_rows=(), qc100_ids=(), *, proportions, seed):
    """Build {question_id: record} from the full run, the layer scan and the old QC sample."""
    qc100_ids = set(qc100_ids)
    _check_indices_are_unique(full_rows, "full")
    _check_indices_are_unique(scan_rows, "scan")
    manifest = {}

    for row in full_rows:
        qid = row["question_id"]
        if qid in manifest:
            raise ValueError(f"{qid}: the full run must have one episode per question")
        if row["source_split"] not in SUPPORTED_SOURCE_SPLITS:
            raise ValueError(f"{qid}: unknown source split {row['source_split']!r}")
        collection_split = row.get("experiment_split", "")
        if collection_split and collection_split not in SUPPORTED_EXPERIMENT_SPLITS:
            raise ValueError(f"{qid}: unknown experiment split {collection_split!r}")
        manifest[qid] = {
            "question_id": qid,
            "source_split": row["source_split"],
            "partition": None,
            "hop_group": hop_group(qid),
            "collection_split": collection_split,
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

    ids_by_source = {}
    for record in manifest.values():
        ids_by_source.setdefault(record["source_split"], []).append(record["question_id"])
    for qid, partition in assign_partitions(ids_by_source, proportions=proportions, seed=seed).items():
        manifest[qid]["partition"] = partition

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
    """Partition sizes, their make-up, and how earlier development data overlaps the held-out sets."""
    records = list(manifest.values())
    in_scan = [m for m in records if m["scan_split"]]
    different = [m for m in in_scan if m["scan_split"] != m["collection_split"]]
    in_qc100 = [m for m in records if m["in_qc100"]]
    in_development = [m for m in records if m["in_development_data"]]

    def by_partition(group):
        return dict(sorted(Counter(m["partition"] for m in group).items()))

    return {
        "questions": len(records),
        "by_partition": by_partition(records),
        "by_partition_source_and_hop": dict(sorted(Counter(
            f"{m['partition']} / {m['source_split']} / {m['hop_group']}" for m in records).items())),
        "scan_questions_also_in_full_run": len(in_scan),
        "scan_questions_with_a_different_collection_split": len(different),
        "qc100_by_partition": by_partition(in_qc100),
        "in_development_data_by_partition": by_partition(in_development),
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
    ``experiment_split`` is ignored; it is a collection-time label, not the partition.
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
        if manifest[qid]["partition"] != allowed:
            raise ValueError(f"{qid}: purpose {purpose!r} requires {allowed} questions")
    return rows


def load_labeled_rows(results_dir, package_dir):
    """Join a run's interactions with the labels and the partition manifest.

    ``results_dir`` holds ``train/`` and ``validation/`` folders, each with an
    ``interactions.jsonl`` (full run or layer scan). ``package_dir`` is the
    folder written by ``scripts/prepare_behavior_v01.py``.

    Returns ``(rows, manifest)``. Each row is the original interaction with two
    additions: ``partition`` and ``label``.
    """
    package_dir = Path(package_dir)
    manifest = read_manifest(package_dir / "split_manifest.csv")
    labels = read_labels(package_dir / "labels.csv")

    rows = read_run_interactions(results_dir)
    for row in rows:
        qid = row["question_id"]
        row["partition"] = manifest[qid]["partition"]
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


DEVELOPMENT_PARTITIONS = ("train", "validation")


def export_partition_activations(rows, manifest, activation_root, *, run, layers,
                                 export_activation_root, export_result_root, export_run_name,
                                 partitions=DEVELOPMENT_PARTITIONS):
    """Write one run's activations and episodes regrouped by partition.

    The output has the layout the collection pipeline already produces, with the
    partition in place of the MuSiQue source split, so existing loaders only
    need a new run name:

        <export_activation_root>/<export_run_name>/layer_NN/<partition>_attempt1.pt
                                                           <partition>_attempt2.pt
                                                           <partition>.pt        (A1 rows, then A2 rows)
        <export_result_root>/<export_run_name>/<partition>/interactions.jsonl  (re-indexed;
            the original row is kept as source_run / source_activation_index)
        <export_result_root>/<export_run_name>/intervention_question_ids.json

    Only the named ``partitions`` are written. The default is the development
    pair, so a training directory never contains ``test.pt`` and nothing can
    score test by accident; ``test`` is exported at the freeze and
    ``intervention`` at the causal stage, each under its own run name. The
    intervention id file is always written. Both destination directories must
    not exist yet, so an export can never overwrite or leave stale files.
    Returns {partition: {"questions", "rows", "files": {path: sha256}}}.
    """
    unknown = set(partitions) - set(PARTITIONS)
    if unknown:
        raise ValueError(f"unknown partitions {sorted(unknown)}; expected a subset of {list(PARTITIONS)}")
    by_partition = {partition: [] for partition in PARTITIONS}
    for row in rows:
        by_partition[manifest[row["question_id"]]["partition"]].append(row)
    empty = [p for p in partitions if not by_partition[p]]
    if empty:
        raise ValueError(f"no rows for partition(s) {empty}; rows and manifest are from different runs")

    activation_dir = Path(export_activation_root) / export_run_name
    result_dir = Path(export_result_root) / export_run_name
    ensure_output_available(activation_dir)
    ensure_output_available(result_dir)
    result_dir.mkdir(parents=True)
    intervention_ids = sorted(row["question_id"] for row in by_partition["intervention"])
    (result_dir / INTERVENTION_IDS_FILE).write_text(json.dumps(intervention_ids, indent=2) + "\n")

    summary = {}
    for partition in PARTITIONS:
        group = by_partition[partition]
        if partition not in partitions:
            summary[partition] = {"questions": len(group), "rows": 0, "files": {}}
            continue
        purpose = PARTITION_PURPOSE[partition]
        files = {}

        for layer in layers:
            attempt1 = load_production_activations(group, manifest, activation_root,
                                                   purpose=purpose, layer=layer, attempt=1, run=run)
            attempt2 = load_production_activations(group, manifest, activation_root,
                                                   purpose=purpose, layer=layer, attempt=2, run=run)
            store = ActivationStore(activation_dir / f"layer_{int(layer):02d}")
            for name, tensor in ((f"{partition}_attempt1", attempt1),
                                 (f"{partition}_attempt2", attempt2),
                                 (partition, torch.cat([attempt1, attempt2], dim=0))):
                _save_and_verify(store, name, tensor)
                path = store._split_path(name)
                files[str(path)] = sha256_file(path)

        records = []
        for position, row in enumerate(group):
            record = dict(row)
            record["partition"] = partition
            record["source_run"] = run
            record["source_activation_index"] = manifest[row["question_id"]][f"{run}_index"]
            record["attempt1_activation_index"] = position
            record["attempt2_activation_index"] = position
            records.append(record)
        records = _add_sae_indices(records, len(records))
        write_jsonl(result_dir / partition / "interactions.jsonl", records)

        summary[partition] = {"questions": len(group), "rows": 2 * len(group), "files": files}
    return summary
