"""Human check of the Behavior v0.1 labels.

Two jobs:

1. Build a blind packet. Annotators see the question, the source paragraphs,
   A1, the Critic's feedback and A2, and none of our labels. A separate private
   key records which episode each row is.
2. Measure how often two annotators agree.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import shutil

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from mas_sae.agents.solver import Solver
from mas_sae.data.musique import SUPPORTED_SOURCE_SPLITS, load_musique_split
from mas_sae.evaluation.behavior_v01 import SOLVER_RESPONSES

# Columns an annotator sees, then the empty columns they fill in.
EVIDENCE_COLUMNS = ["review_id", "question", "context", "solver_a1", "critic_feedback", "solver_a2"]
ANSWER_COLUMNS = ["feedback_type", "solver_response", "eligible_primary",
                  "annotator_confidence", "notes"]

# The values an annotator may enter.
ALLOWED_ANSWERS = {
    "feedback_type": {"answer", "refusal", "premise_rejection", "unresolved"},
    "solver_response": set(SOLVER_RESPONSES),
    "eligible_primary": {"yes", "no", "uncertain"},
}

ROWS_PER_GROUP = 40          # sampled from each group, except 'kept A1' which is taken whole
SECOND_ANNOTATOR_ROWS = 10   # per group, also given to a second annotator


# ---------------------------------------------------------------- building the packet

def qc_stratum(row):
    """Group used only to sample the packet. Annotators never see it."""
    label = row["label"]
    if label["feedback_type"] != "answer":
        return label["feedback_type"]
    if label["a1_kind"] != "answer":
        return "a1_nonanswer_or_unresolved"
    if label["conflict_status"] == "containment":
        return "containment"
    if label["eligible_primary"]:
        return label["solver_response"]
    return "other_" + label["solver_response"]


def source_paragraphs(question_ids, dataset_revision):
    """The MuSiQue paragraphs the Solver was given, as {question_id: paragraphs}."""
    wanted = set(question_ids)
    paragraphs = {}
    for source_split in sorted(SUPPORTED_SOURCE_SPLITS):
        for example in load_musique_split(source_split, revision=dataset_revision):
            if example["id"] in wanted:
                paragraphs[example["id"]] = example["paragraphs"]
    return paragraphs


def choose_qc_rows(rows, seed):
    """Pick discovery rows for the packet and decide which go to a second annotator.

    Every 'kept A1' case is taken because there are so few; other groups
    contribute a fixed number each. Returns (chosen rows, private key, sampling record).
    """
    groups = defaultdict(list)
    for row in rows:
        if row["canonical_split"] == "discovery":
            groups[qc_stratum(row)].append(row)

    rng = random.Random(seed)
    chosen = []
    for name, members in sorted(groups.items()):
        members = sorted(members, key=lambda r: r["question_id"])
        if name == "retained_a1":
            chosen += members
        else:
            chosen += rng.sample(members, min(ROWS_PER_GROUP, len(members)))
    rng.shuffle(chosen)

    positions = defaultdict(list)
    for position, row in enumerate(chosen):
        positions[qc_stratum(row)].append(position)
    second = set()
    for group_positions in positions.values():
        count = min(SECOND_ANNOTATOR_ROWS, len(group_positions))
        second.update(rng.sample(group_positions, count))

    key = [{"review_id": f"Q{position + 1:04d}",
            "question_id": row["question_id"],
            "stratum": qc_stratum(row),
            "second_annotator": position in second}
           for position, row in enumerate(chosen)]
    sampling = {
        "seed": seed,
        "strata_rule_version": rows[0]["label"]["rule_version"],
        "population": {name: len(members) for name, members in sorted(groups.items())},
        "selected": dict(sorted(Counter(entry["stratum"] for entry in key).items())),
        "second_annotator_rows": len(second),
    }
    return chosen, key, sampling


def write_annotator_files(qc_dir, chosen, key, paragraphs):
    """Write annotator_a.csv (every row) and annotator_b.csv (the second annotator's rows)."""
    shown = []
    for row, entry in zip(chosen, key):
        shown.append({
            "review_id": entry["review_id"],
            "question": row["question"],
            "context": Solver.format_paragraphs(paragraphs[row["question_id"]]),
            "solver_a1": row["solver_attempt_1"],
            "critic_feedback": row["critic_feedback"],
            "solver_a2": row["solver_attempt_2"],
        })
    for_second = [s for s, entry in zip(shown, key) if entry["second_annotator"]]

    for name, rows_shown in (("annotator_a.csv", shown), ("annotator_b.csv", for_second)):
        with open(Path(qc_dir) / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVIDENCE_COLUMNS + ANSWER_COLUMNS)
            writer.writeheader()
            writer.writerows(rows_shown)


def reuse_qc_packet(previous_qc_dir, qc_dir):
    """Copy an already issued packet unchanged, so files given to annotators stay valid.

    Returns (private key, sampling record).
    """
    previous_qc_dir, qc_dir = Path(previous_qc_dir), Path(qc_dir)
    for name in ("annotator_a.csv", "annotator_b.csv"):
        shutil.copyfile(previous_qc_dir / name, qc_dir / name)

    old_key = json.loads((previous_qc_dir / "private_key.json").read_text())
    old_sampling = json.loads((previous_qc_dir / "sampling.json").read_text())

    key = [{"review_id": entry["review_id"],
            "question_id": entry["question_id"],
            "stratum": entry["stratum"],
            "second_annotator": entry["second_annotator"]} for entry in old_key]

    # The groups were formed under the rule version in force when the packet was sampled.
    strata_rule_version = (old_sampling.get("strata_rule_version")
                           or old_key[0]["candidate"]["rule_version"])
    sampling = {
        "seed": old_sampling["seed"],
        "strata_rule_version": strata_rule_version,
        "population": old_sampling["population"],
        "selected": old_sampling["selected"],
        "second_annotator_rows": sum(entry["second_annotator"] for entry in key),
        "reused_from": str(previous_qc_dir),
    }
    return key, sampling


def _check_annotator_file(path, entries, rows_by_question):
    """One annotator file must list exactly ``entries``, match the run's text and start blank."""
    shown = load_annotations(path)
    if list(shown) != [entry["review_id"] for entry in entries]:
        raise ValueError(f"{path.name} and the private key disagree on which rows it holds")

    for entry in entries:
        review_id = entry["review_id"]
        row_shown = shown[review_id]
        row = rows_by_question[entry["question_id"]]
        if list(row_shown) != EVIDENCE_COLUMNS + ANSWER_COLUMNS:
            raise ValueError(f"{path.name} has unexpected columns")
        text_shown = (row_shown["question"], row_shown["solver_a1"],
                      row_shown["critic_feedback"], row_shown["solver_a2"])
        text_in_run = (row["question"], row["solver_attempt_1"],
                       row["critic_feedback"], row["solver_attempt_2"])
        if row["canonical_split"] != "discovery":
            raise ValueError(f"{review_id}: QC row is not a discovery question")
        if text_shown != text_in_run:
            raise ValueError(f"{review_id} in {path.name}: text differs from the production run")
        if any(row_shown[column] for column in ANSWER_COLUMNS):
            raise ValueError(f"{review_id} in {path.name}: annotation columns must start empty")


def check_qc_packet(qc_dir, key, rows_by_question):
    """Both annotator files must be discovery-only, match the run's text, and show no labels."""
    qc_dir = Path(qc_dir)
    for_second = [entry for entry in key if entry["second_annotator"]]
    _check_annotator_file(qc_dir / "annotator_a.csv", key, rows_by_question)
    _check_annotator_file(qc_dir / "annotator_b.csv", for_second, rows_by_question)


# ---------------------------------------------------------------- scoring agreement

def load_annotations(path):
    """Read an annotator file as {review_id: row}, keeping the file's row order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {row["review_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate review ID")
    return by_id


def _entered(row, field):
    return row.get(field, "").strip()


def agreement(a, b):
    """Compare two {review_id: row} annotation sets, one column at a time.

    Only rows both annotators completed are compared. Neither annotator is
    treated as the truth; disagreements are listed so they can be resolved.
    """
    for annotations in (a, b):
        for row in annotations.values():
            for field, allowed in ALLOWED_ANSWERS.items():
                value = _entered(row, field)
                if value and value not in allowed:
                    raise ValueError(f"Invalid {field}: {value!r}")

    shared = sorted(a.keys() & b.keys())
    for review_id in shared:
        for field in EVIDENCE_COLUMNS[1:]:
            if a[review_id].get(field) != b[review_id].get(field):
                raise ValueError(
                    f"Different episode evidence under shared review ID {review_id}")

    result = {"common_review_ids": len(shared), "fields": {}}
    for field in ALLOWED_ANSWERS:
        completed = [i for i in shared if _entered(a[i], field) and _entered(b[i], field)]
        from_a = [_entered(a[i], field) for i in completed]
        from_b = [_entered(b[i], field) for i in completed]
        labels = sorted(set(from_a + from_b))
        matches = sum(x == y for x, y in zip(from_a, from_b))

        kappa = None
        if len(labels) > 1:
            kappa = float(cohen_kappa_score(from_a, from_b))
        matrix = []
        if completed:
            matrix = confusion_matrix(from_a, from_b, labels=labels).tolist()

        result["fields"][field] = {
            "paired_completed": len(completed),
            "raw_agreement": matches / len(completed) if completed else None,
            "cohen_kappa": kappa,
            "labels": labels,
            "confusion_matrix": matrix,
            "disagreement_ids": [i for i, x, y in zip(completed, from_a, from_b) if x != y],
        }
    return result
