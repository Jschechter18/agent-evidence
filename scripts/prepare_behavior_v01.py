"""Build the Behavior v0.1.1 label package from the finished production run.

Writes four things into a new folder:

    labels.csv           one row per question: the new labels next to lexical_v2
    split_manifest.csv   one row per question: which partition it belongs to
    counts.json          label counts and the split audit
    qc/                  blind annotation packet for the human check

The collection run folders are only read, never changed.

    PYTHONPATH=src python scripts/prepare_behavior_v01.py \\
        --artifacts /home/ubuntu/capstone-artifacts \\
        --output /home/ubuntu/capstone-artifacts/<new folder> \\
        --reuse-qc <existing package>/qc

Without --reuse-qc a new QC packet is sampled.
"""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from mas_sae.data.production import (
    canonical_manifest,
    check_scan_repeats_full_run,
    read_run_interactions,
    split_audit,
    write_manifest,
)
from mas_sae.evaluation.behavior_qc import (
    check_qc_packet,
    choose_qc_rows,
    reuse_qc_packet,
    source_paragraphs,
    write_annotator_files,
)
from mas_sae.evaluation import behavior, behavior_v01, scoring
from mas_sae.evaluation.behavior_v01 import classify_candidate, label_counts, write_labels
from mas_sae.experiments.artifacts import (
    ensure_output_available,
    get_git_commit,
    get_git_diff_sha256,
    sha256_file,
    write_json_atomic,
)

# Where the finished runs live inside the artifacts folder.
FULL_RESULTS = "natural_4b_full/results"
SCAN_RESULTS = "issue69_natural_4b_layer_scan/results/collection/natural_4b_layer_scan"
FIRST_QC_SAMPLE = "issue55_human_qc/sample_manifest.json"
QC_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "qc_guide.md"

# The modules whose code decides the labels; their hashes go into counts.json.
RULE_MODULES = (behavior_v01, behavior, scoring)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="folder holding the collection runs")
    parser.add_argument("--output", type=Path, required=True,
                        help="new folder for the package")
    parser.add_argument("--reuse-qc", type=Path,
                        help="qc/ folder of an issued packet to keep unchanged")
    parser.add_argument("--seed", type=int, default=42,
                        help="used only when a new QC packet is sampled")
    args = parser.parse_args()
    ensure_output_available(args.output)

    full_dir = args.artifacts / FULL_RESULTS
    scan_dir = args.artifacts / SCAN_RESULTS

    # Remember the input files' hashes so we can confirm they were not changed.
    input_files = [run_dir / source_split / "interactions.jsonl"
                   for run_dir in (full_dir, scan_dir)
                   for source_split in ("train", "validation")]
    hashes_before = {str(path): sha256_file(path) for path in input_files}

    # 1. Read both runs and decide each question's partition.
    full_rows = read_run_interactions(full_dir)
    scan_rows = read_run_interactions(scan_dir)
    check_scan_repeats_full_run(full_rows, scan_rows)

    first_qc_sample = json.loads((args.artifacts / FIRST_QC_SAMPLE).read_text())
    qc100_ids = [sample["question_id"] for sample in first_qc_sample["samples"]]
    manifest = canonical_manifest(full_rows, scan_rows, qc100_ids)

    # 2. Label every episode of the full run.
    for row in full_rows:
        row["canonical_split"] = manifest[row["question_id"]]["canonical_split"]
        row["label"] = classify_candidate(row)

    # 3. Write the two tables and the counts.
    args.output.mkdir(parents=True)
    write_labels(args.output / "labels.csv", full_rows)
    write_manifest(args.output / "split_manifest.csv", manifest)

    train_rows = [row for row in full_rows if row["source_split"] == "train"]
    validation_rows = [row for row in full_rows if row["source_split"] == "validation"]
    counts = {
        "rule_version": full_rows[0]["label"]["rule_version"],
        "code": {
            "git_commit": get_git_commit(),
            "uncommitted_changes_sha256": get_git_diff_sha256(),
            "rule_files_sha256": {Path(module.__file__).name: sha256_file(module.__file__)
                                  for module in RULE_MODULES},
        },
        "source_files_sha256": hashes_before,
        "all": label_counts(full_rows),
        "source_train": label_counts(train_rows),
        "source_validation": label_counts(validation_rows),
        "split_audit": split_audit(manifest),
    }
    write_json_atomic(args.output / "counts.json", counts)

    # 4. The blind QC packet: reuse the issued one, or sample a new one.
    qc_dir = args.output / "qc"
    qc_dir.mkdir()
    if args.reuse_qc:
        key, sampling = reuse_qc_packet(args.reuse_qc, qc_dir)
    else:
        run_config = yaml.safe_load((full_dir / "train" / "resolved_config.yaml").read_text())
        chosen, key, sampling = choose_qc_rows(full_rows, args.seed)
        paragraphs = source_paragraphs([row["question_id"] for row in chosen],
                                       run_config["dataset"]["revision"])
        write_annotator_files(qc_dir, chosen, key, paragraphs)

    rows_by_question = {row["question_id"]: row for row in full_rows}
    check_qc_packet(qc_dir, key, rows_by_question)

    # The private key also holds our own labels, to score the annotations later.
    for entry in key:
        row = rows_by_question[entry["question_id"]]
        entry["lexical_v2_label"] = row["solver_behavior"]
        entry["feedback_type"] = row["label"]["feedback_type"]
        entry["solver_response"] = row["label"]["solver_response"]
        entry["eligible_primary"] = row["label"]["eligible_primary"]
    (qc_dir / "private_key.json").write_text(json.dumps(key, indent=2) + "\n")
    write_json_atomic(qc_dir / "sampling.json", sampling)
    shutil.copyfile(QC_GUIDE, qc_dir / "INSTRUCTIONS.md")

    # 5. Confirm the inputs are untouched and print a short summary.
    if {str(path): sha256_file(path) for path in input_files} != hashes_before:
        raise RuntimeError("A source file changed while the package was being built")
    summary = {
        "output": str(args.output),
        "primary_target_by_split": counts["all"]["primary_target_by_split"],
        "split_audit": counts["split_audit"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
