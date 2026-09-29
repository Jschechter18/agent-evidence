"""
Activation-layer selection: use a linear probe on real Solver-Critic Attempt-2
activations to decide WHICH layer is most predictive of solver_accepted_feedback,
before any SAE gets trained specifically for downstream analysis.

Uses the real train/validation split produced by the collection pipeline
directly (results/collection/<run_name>/{train,validation}/interactions.jsonl),
rather than a random split.

Config is loaded from a YAML file (default: configs/activation_layer_selection.yaml)
via --config, rather than hardcoded constants, so a new run (e.g. once #36's
scaled collection lands) only requires editing the YAML, not the code.

NOTE: as of this writing, run_name="scale_smoke_1q" is a 1-question smoke
test. This run validates that the loading/probing/comparison mechanics work correctly
end-to-end on real data; re-run against a larger collection run before
trusting the recommended layer.

"""

import argparse
import json
from pathlib import Path

from mas_sae.probe.config import ActivationLayerSelectionConfig
from mas_sae.probe.data import load_real_layer_split
from mas_sae.probe.model import fit_probe
from mas_sae.utils.versioning import create_versioned_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/activation_layer_selection.yaml"),
        help="Path to a YAML config file (see ActivationLayerSelectionConfig).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ActivationLayerSelectionConfig.from_yaml(args.config)
    run_dir = create_versioned_run_dir(config.output_dir)

    results = {}
    for layer in config.candidate_layers:
        X_train, y_train = load_real_layer_split(
            config.run_name, "train", layer, config.data_root, config.results_root
        )
        X_val, y_val = load_real_layer_split(
            config.run_name, "validation", layer, config.data_root, config.results_root
        )

        print(f"layer_{layer}: {X_train.shape[0]} train episodes, {X_val.shape[0]} val episodes")

        try:
            probe, scaler, metrics, *_ = fit_probe(X_train, y_train, X_val, y_val, seed=config.seed)
            metrics.pop("f1", None)  # excluded 
            metrics["layer_number"] = int(layer)  # programmatic access without re-parsing the key string
            results[f"layer_{layer}"] = metrics
            print(f"  metrics: {metrics}")
        except ValueError as e:
            results[f"layer_{layer}"] = {"error": str(e), "layer_number": int(layer)}
            print(f"  could not fit/score probe on this split: {e}")

    scored = {k: v for k, v in results.items() if "auroc" in v}
    if scored:
        ranked = sorted(scored.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
        recommended_layer = ranked[0][0]
        print(f"\nRecommended layer: {recommended_layer} (by validation AUROC)")
    else:
        ranked = []
        recommended_layer = None
        print(
            "\nNo layer produced a valid AUROC on this run -- expected on a "
            "dataset this small. Re-run against a larger collection run."
        )

    with open(run_dir / f"{config.run_name}_layer_selection_results.json", "w") as f:
        json.dump(
            {
                "run_name": config.run_name,
                "results_by_layer": results,
                "ranked_layers": [name for name, _ in ranked],
                "recommended_layer": recommended_layer,
            },
            f,
            indent=2,
        )

    print(f"\nAll layer-selection outputs written to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
