from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.experiments.natural_validation import (
    ARM_ROLES,
    SOLVER_ROLE,
    load_manifest,
    load_validation_config,
    load_validation_examples,
    run_natural_validation,
)
from mas_sae.models.roles import configure_agent, load_role_models


logger = logging.getLogger(__name__)
RESULT_ROOT = Path("results/natural_validation")


def parse_args() -> argparse.Namespace:
    """Parse Natural-critic validation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate the Natural blind-first critic on paired same-model and "
                    "cross-model arms from a pinned YAML config (no activations, no Validator)."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to a validation YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/natural_validation/<run_name>).")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing output directory; completed rows are reused, never regenerated.")
    return parser.parse_args()


def main() -> None:
    """Run one config-driven Natural-critic validation."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_validation_config(args.config)
    output_dir = args.output_dir or RESULT_ROOT / config["output"]["run_name"]
    manifest, manifest_sha256 = load_manifest(config["manifest"]["path"], config["manifest"]["sha256"])
    logger.info("Manifest %s verified (%d questions, sha256 %s)",
                config["manifest"]["path"], len(manifest), manifest_sha256)

    examples = load_validation_examples(config, manifest)
    logger.info("Loaded %d questions from %s@%s split=%s", len(examples), config["dataset"]["repo"],
                config["dataset"]["revision"], config["dataset"]["source_split"])

    loaded = load_role_models(config["roles"])
    solver = Solver(*loaded[SOLVER_ROLE])
    critics = {arm: Critic(*loaded[role], blind_then_compare=True) for arm, role in ARM_ROLES.items()}
    agents = {SOLVER_ROLE: solver, **{role: critics[arm] for arm, role in ARM_ROLES.items()}}
    for role, agent in agents.items():
        configure_agent(agent, config["roles"][role])

    summary = run_natural_validation(config=config, output_dir=output_dir, solver=solver, critics=critics,
                                     agents=agents, examples=examples, manifest=manifest,
                                     manifest_sha256=manifest_sha256, resume=args.resume)
    for arm, counts in summary["arms"].items():
        logger.info("Arm %s: completed=%d failed=%d", arm, counts["n_completed"], counts["n_failed"])
    logger.info("Wrote %s", output_dir)


if __name__ == "__main__":
    main()
