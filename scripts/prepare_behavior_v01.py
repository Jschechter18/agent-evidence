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
        --train-proportion 0.8 --test-proportion 0.1 --intervention-proportion 0.1 \\
        --split-seed 42

The partition rule: every MuSiQue-validation question is ``validation``;
MuSiQue-train questions are cut by question id, stratified by hop count, into
train / test / intervention with the proportions given here. The team agreed
0.8 / 0.1 / 0.1 with seed 42 on 2026-10-06. That cut was made once; the result
is the committed ``partitions.csv`` and every later run keeps it:

    PYTHONPATH=src python scripts/prepare_behavior_v01.py \\
        --full-results <artifacts>/<rerun>/results \\
        --output <artifacts>/<new folder> \\
        --frozen-partitions results/behavior/behavior_v011_split80_10_10_20261006/partitions.csv \\
        --coverage complete

With --frozen-partitions membership is read from the file and never recomputed;
only the activation indices are rebuilt for the new run. A question the file
does not know stops the run. A run that deliberately covers part of the
question set, such as a layer scan, passes --coverage subset and the uncovered
ids are written next to the package.

The QC packet is sampled from the train partition only. Pass --reuse-qc only
for a packet that was itself sampled under the same partition rule; a packet
issued under an earlier split is refused because it would show annotators
test or intervention questions.
"""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from mas_sae.data.production import (
    COVERAGE_MODES,
    build_partition_manifest,
    check_scan_repeats_full_run,
    read_partitions,
    read_run_interactions,
    split_audit,
    write_package_tables,
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

# Where the first production run, the layer scan and the first QC sample live
# inside the artifacts folder. Defaults only; any run can be passed explicitly.
FULL_RESULTS = "natural_4b_full/results"
SCAN_RESULTS = "issue69_natural_4b_layer_scan/results/collection/natural_4b_layer_scan"
FIRST_QC_SAMPLE = "issue55_human_qc/sample_manifest.json"
QC_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "qc_guide.md"

# The modules whose code decides the labels; their hashes go into counts.json.
RULE_MODULES = (behavior_v01, behavior, scoring)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path,
                        help="folder holding the collection runs; gives the historical defaults for "
                             "--full-results, --scan-results and --first-qc-sample")
    parser.add_argument("--full-results", type=Path,
                        help="the run's results folder, with train/ and validation/ interactions.jsonl "
                             f"(default: <artifacts>/{FULL_RESULTS})")
    parser.add_argument("--scan-results", type=Path,
                        help="a layer scan's results folder to record alongside the run "
                             f"(default: <artifacts>/{SCAN_RESULTS} when that folder exists)")
    parser.add_argument("--first-qc-sample", type=Path,
                        help="sample_manifest.json of the first 100-row QC sample "
                             f"(default: <artifacts>/{FIRST_QC_SAMPLE} when that file exists)")
    parser.add_argument("--output", type=Path, required=True,
                        help="new folder for the package")
    parser.add_argument("--reuse-qc", type=Path,
                        help="qc/ folder of an issued packet to keep unchanged")
    parser.add_argument("--seed", type=int, default=42,
                        help="used only when a new QC packet is sampled")
    parser.add_argument("--frozen-partitions", type=Path,
                        help="partitions.csv to take every question's partition from, never recomputing "
                             "membership; use this for every run after the first")
    parser.add_argument("--coverage", choices=COVERAGE_MODES, default="complete",
                        help="with --frozen-partitions: 'complete' fails if any frozen question is "
                             "missing from the run; 'subset' records the missing ids instead")
    parser.add_argument("--train-proportion", type=float,
                        help="first derivation only: share of MuSiQue-train questions that become train")
    parser.add_argument("--test-proportion", type=float,
                        help="first derivation only: share of MuSiQue-train questions that become test")
    parser.add_argument("--intervention-proportion", type=float,
                        help="first derivation only: share reserved for the causal experiment")
    parser.add_argument("--split-seed", type=int,
                        help="first derivation only: seed of the hop-stratified cut")
    args = parser.parse_args()

    deriving = args.frozen_partitions is None
    derivation = (args.train_proportion, args.test_proportion, args.intervention_proportion, args.split_seed)
    if deriving and any(value is None for value in derivation):
        parser.error("without --frozen-partitions, --train-proportion, --test-proportion, "
                     "--intervention-proportion and --split-seed are all required")
    if not deriving and any(value is not None for value in derivation):
        parser.error("--frozen-partitions keeps the partitions as they are; do not pass proportions or a seed")
    if args.full_results is None and args.artifacts is None:
        parser.error("pass --full-results, or --artifacts for the historical layout")

    full_dir = args.full_results or args.artifacts / FULL_RESULTS
    scan_dir = args.scan_results
    if scan_dir is None and args.artifacts is not None and (args.artifacts / SCAN_RESULTS).is_dir():
        scan_dir = args.artifacts / SCAN_RESULTS
    qc_sample = args.first_qc_sample
    if qc_sample is None and args.artifacts is not None and (args.artifacts / FIRST_QC_SAMPLE).is_file():
        qc_sample = args.artifacts / FIRST_QC_SAMPLE
    ensure_output_available(args.output)

    # Remember the input files' hashes so we can confirm they were not changed.
    input_files = [run_dir / source_split / "interactions.jsonl"
                   for run_dir in (full_dir, scan_dir) if run_dir is not None
                   for source_split in ("train", "validation")]
    hashes_before = {str(path): sha256_file(path) for path in input_files}

    # 1. Read the run(s) and give each question its partition.
    full_rows = read_run_interactions(full_dir)
    scan_rows = read_run_interactions(scan_dir) if scan_dir is not None else []
    if scan_rows:
        check_scan_repeats_full_run(full_rows, scan_rows)
    qc100_ids = []
    if qc_sample is not None:
        qc100_ids = [sample["question_id"] for sample in json.loads(qc_sample.read_text())["samples"]]

    if deriving:
        proportions = {"train": args.train_proportion, "test": args.test_proportion,
                       "intervention": args.intervention_proportion}
        frozen = None
        manifest = build_partition_manifest(full_rows, scan_rows, qc100_ids,
                                            proportions=proportions, seed=args.split_seed)
    else:
        frozen = read_partitions(args.frozen_partitions)
        manifest = build_partition_manifest(full_rows, scan_rows, qc100_ids,
                                            frozen=frozen, coverage=args.coverage)

    # 2. Label every episode of the full run.
    for row in full_rows:
        row["partition"] = manifest[row["question_id"]]["partition"]
        row["label"] = classify_candidate(row)

    # 3. Write the tables and the counts. partitions.csv is the frozen mapping
    #    in full whenever one is in use, however much of it this run covers.
    args.output.mkdir(parents=True)
    write_labels(args.output / "labels.csv", full_rows)
    coverage = write_package_tables(args.output, manifest, frozen)
    if deriving:
        partition_record = {"proportions_of_source_train": proportions, "seed": args.split_seed,
                            "rule": "MuSiQue-validation -> validation; MuSiQue-train cut by sorted "
                                    "question id, stratified by hop group"}
    else:
        partition_record = {"frozen_partitions": str(args.frozen_partitions),
                            "frozen_partitions_sha256": sha256_file(args.frozen_partitions),
                            "coverage": args.coverage,
                            "questions": {k: coverage[k] for k in ("expected", "collected", "missing")},
                            "rule": "membership taken from the frozen mapping; only activation "
                                    "indices were rebuilt for this run"}

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
        "inputs": {"full_results": str(full_dir),
                   "scan_results": None if scan_dir is None else str(scan_dir),
                   "first_qc_sample": None if qc_sample is None else str(qc_sample)},
        "partition": partition_record,
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
