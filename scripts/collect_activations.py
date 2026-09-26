from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.experiments.artifacts import ensure_output_available
from mas_sae.experiments.collection import (
    collect_examples,
    select_questions,
)
from mas_sae.experiments.collection_artifacts import (
    build_resolved_config,
    save_collection_artifacts,
)
from mas_sae.experiments.collection_config import load_collection_config
from mas_sae.models.loader import load_gemma


logger = logging.getLogger(__name__)
RESULT_ROOT = Path("results/collection")
ACTIVATION_ROOT = Path("data/activations")

# What each ``collection.protocol_version`` label selects. The package
# never sees these names; it only receives the capabilities below.
PROTOCOLS: dict[str, dict[str, bool]] = {
    "v1": {
        "blind_then_compare": False,
        "controlled_as_own_conclusion": False,
        "type_checked_target": False,
    },
    "v2": {
        "blind_then_compare": True,
        "controlled_as_own_conclusion": True,
        "type_checked_target": True,
    },
}
DEFAULT_PROTOCOL = "v1"


def parse_args() -> argparse.Namespace:
    """Parse activation-collection command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect Solver-Critic activations from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a collection YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one config-driven activation collection."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_collection_config(args.config)
    model_id = config["model"]["id"]
    source_split = config["dataset"]["source_split"]
    num_questions = config["dataset"]["num_questions"]
    layers = config["collection"]["layers"]
    seed = config["collection"]["seed"]
    protocol_version = config["collection"].get(
        "protocol_version", DEFAULT_PROTOCOL
    )
    run_name = config["output"]["run_name"]

    if protocol_version not in PROTOCOLS:
        raise ValueError(
            "collection.protocol_version must be one of "
            f"{list(PROTOCOLS)}, got {protocol_version!r}."
        )
    protocol = PROTOCOLS[protocol_version]

    ensure_output_available(RESULT_ROOT / run_name / source_split)
    candidate_sites = [
        f"model.language_model.layers.{layer}" for layer in layers
    ]

    logger.info(
        "Starting run=%s split=%s questions=%d protocol=%s",
        run_name,
        source_split,
        num_questions,
        protocol_version,
    )
    logger.info("Loading model %s", model_id)

    model, processor = load_gemma(model_id=model_id)
    solver = Solver(model, processor)
    critic = Critic(
        model,
        processor,
        blind_then_compare=protocol["blind_then_compare"],
        controlled_as_own_conclusion=protocol["controlled_as_own_conclusion"],
    )
    validator = Validator(model, processor)

    resolved_config = build_resolved_config(
        config,
        model,
        solver,
        critic,
        validator,
        protocol_version=protocol_version,
        type_checked_target=protocol["type_checked_target"],
    )
    selection = select_questions(config["dataset"], default_seed=seed)

    if selection["sampled_questions"] is not None:
        logger.info(
            "Sampled %d questions (strategy=%s, experiment_split=%s)",
            len(selection["examples"]),
            config["dataset"].get("sampling", {}).get("strategy", "first_n"),
            "yes" if selection["experiment_splits"] is not None else "no",
        )

    result = collect_examples(
        examples=selection["examples"],
        source_split=source_split,
        model=model,
        solver=solver,
        critic=critic,
        validator=validator,
        candidate_sites=candidate_sites,
        base_seed=seed,
        experiment_splits=selection["experiment_splits"],
        type_checked_target=protocol["type_checked_target"],
    )

    summary = save_collection_artifacts(
        activation_root=ACTIVATION_ROOT,
        result_root=RESULT_ROOT,
        run_name=run_name,
        source_split=source_split,
        candidate_sites=candidate_sites,
        attempt1_by_site=result["attempt1_by_site"],
        attempt2_by_site=result["attempt2_by_site"],
        records=result["records"],
        resolved_config=resolved_config,
        sampled_questions=selection["sampled_questions"],
    )

    logger.info(
        "Completed run=%s split=%s questions=%d episodes=%d",
        run_name,
        source_split,
        summary["num_questions"],
        summary["num_episodes"],
    )


if __name__ == "__main__":
    main()
