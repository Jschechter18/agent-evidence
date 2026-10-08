"""Materialise one collected run's activations by partition, for the SAE and probe loaders.

Reads the frozen partition manifest from a Behavior package and the run's
existing tensors, and writes the collection layout under a new run name with
the partition in place of the MuSiQue source split:

    <export-activation-root>/<export-run-name>/layer_NN/{train,validation,test}[_attemptN].pt
    <export-result-root>/<export-run-name>/<partition>/interactions.jsonl
    <export-result-root>/<export-run-name>/intervention_question_ids.json
    <export-result-root>/<export-run-name>/export_summary.json

Nothing is recomputed and no source file is changed. The destination must not
exist. Which partitions are written is stated explicitly on the command line.
Before anything is written, every question's partition in the package manifest
is checked against the committed frozen mapping (``--frozen-partitions``); a
package built under a different assignment is refused.

    PYTHONPATH=src python scripts/export_partition_activations.py \\
        --package /home/ubuntu/capstone-artifacts/<package folder> \\
        --source-results /home/ubuntu/capstone-artifacts/natural_4b_full/results \\
        --source-activations /home/ubuntu/capstone-artifacts/natural_4b_full/data \\
        --run full --layers 8 17 25 33 \\
        --partitions train validation test \\
        --export-run-name natural_4b_partitioned \\
        --export-activation-root <export root>/data/activations \\
        --export-result-root <export root>/results/collection

For the nine-layer scan pass its results and activation folders with --run scan.
"""
import argparse
import json
from pathlib import Path

from mas_sae.data.production import (
    PARTITIONS,
    PARTITIONS_FILE,
    check_manifest_matches_frozen,
    export_partition_activations,
    load_labeled_rows,
    read_partitions,
)
from mas_sae.experiments.artifacts import (
    get_git_commit,
    get_git_diff_sha256,
    sha256_file,
    write_json_atomic,
)

# The frozen question-to-partition mapping this repository's runs are bound to.
FROZEN_PARTITIONS = (Path(__file__).resolve().parents[1] / "results" / "behavior"
                     / "behavior_v011_split80_10_10_20261006" / PARTITIONS_FILE)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True,
                        help="Behavior package folder holding split_manifest.csv, labels.csv, counts.json")
    parser.add_argument("--source-results", type=Path, required=True,
                        help="run folder with train/ and validation/ interactions.jsonl")
    parser.add_argument("--source-activations", type=Path, required=True,
                        help="run folder with layer_NN/ activation tensors")
    parser.add_argument("--run", choices=("full", "scan"), required=True,
                        help="which run the source folders are; decides the manifest index used")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS, required=True,
                        help="partitions to write; the intervention id file is always written")
    parser.add_argument("--export-run-name", required=True)
    parser.add_argument("--export-activation-root", type=Path, required=True)
    parser.add_argument("--export-result-root", type=Path, required=True)
    parser.add_argument("--frozen-partitions", type=Path, default=FROZEN_PARTITIONS,
                        help="committed partitions.csv the package manifest must agree with "
                             "(default: the repository's frozen mapping)")
    args = parser.parse_args()

    rows, manifest = load_labeled_rows(args.source_results, args.package)
    check_manifest_matches_frozen(manifest, read_partitions(args.frozen_partitions))
    summary = export_partition_activations(
        rows, manifest, args.source_activations, run=args.run, layers=args.layers,
        export_activation_root=args.export_activation_root,
        export_result_root=args.export_result_root,
        export_run_name=args.export_run_name, partitions=args.partitions)

    package_counts = json.loads((args.package / "counts.json").read_text())
    provenance = {
        "git_commit": get_git_commit(),
        "uncommitted_changes_sha256": get_git_diff_sha256(),
        "package": str(args.package),
        "manifest_sha256": sha256_file(args.package / "split_manifest.csv"),
        "labels_sha256": sha256_file(args.package / "labels.csv"),
        "frozen_partitions": str(args.frozen_partitions),
        "frozen_partitions_sha256": sha256_file(args.frozen_partitions),
        "partition_rule": package_counts.get("partition"),
        "source_run": args.run,
        "source_results": str(args.source_results),
        "source_activations": str(args.source_activations),
        "export_run_name": args.export_run_name,
        "layers": args.layers,
        "partitions_written": list(args.partitions),
        "partition_counts": {p: {"questions": s["questions"], "rows": s["rows"]} for p, s in summary.items()},
        "file_sha256": {path: digest for s in summary.values() for path, digest in s["files"].items()},
    }
    write_json_atomic(args.export_result_root / args.export_run_name / "export_summary.json", provenance)
    print(json.dumps({k: provenance[k] for k in ("export_run_name", "partitions_written", "partition_counts")}, indent=2))


if __name__ == "__main__":
    main()
