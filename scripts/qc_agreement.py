"""Measure agreement between two annotators on the blind QC packet.

    PYTHONPATH=src python scripts/qc_agreement.py annotator_a.csv annotator_b.csv \\
        --output agreement.json

Only rows that both annotators completed are compared. Disagreements are
listed by review ID so they can be resolved.
"""
import argparse
from pathlib import Path

from mas_sae.evaluation.behavior_qc import agreement, load_annotations
from mas_sae.experiments.artifacts import ensure_output_available, write_json_atomic


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="first annotator's completed CSV")
    parser.add_argument("b", type=Path, help="second annotator's completed CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ensure_output_available(args.output)
    result = agreement(load_annotations(args.a), load_annotations(args.b))
    write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
