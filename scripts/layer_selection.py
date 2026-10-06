"""Layer selection under the 6 Oct split contract (development partitions only).

    python scripts/layer_selection.py --config configs/layer_selection_full.yaml --inspect   # FIRST
    python scripts/layer_selection.py --config configs/layer_selection_full.yaml             # holdout: fit train, rank validation
    python scripts/layer_selection.py --config configs/layer_selection_scan.yaml             # cv over train+validation (nine layers)
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from mas_sae.probe.layer_selection import LayerSelectionConfig, inspect_inputs, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/layer_selection_full.yaml"))
    parser.add_argument("--inspect", action="store_true",
                        help="print schemas of labels/manifest/interactions + alignment check, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = LayerSelectionConfig.from_yaml(args.config)

    if args.inspect:
        inspect_inputs(cfg)
        return

    run_dir = run(cfg)
    df = pd.read_csv(run_dir / "layer_comparison.csv")
    show = ["name", "balanced_accuracy", "ci95_lo", "ci95_hi", "within_noise_of_best", "median_chosen_C"]
    print("\n" + df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    comp = json.loads((run_dir / "layer_selection_summary.json").read_text())["comparison"]
    if comp["recommended_layer"]:
        print(f"\nRecommended layer: {comp['recommended_layer']}")
    else:
        print(f"\nNO RECOMMENDATION ({comp['no_recommendation_reason']}). Top by balanced accuracy: "
              f"{comp['top_layer_by_balanced_accuracy']}; tied with: {comp['tied_with_top'] or 'none'}")
    print(f"Layers indistinguishable from the top: {comp['layers_indistinguishable_from_best']}")
    print(f"Full results: {run_dir}")


if __name__ == "__main__":
    main()
