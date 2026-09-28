from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.data.musique import describe_sampled_questions
from mas_sae.experiments import collection_progress
from mas_sae.experiments.collection import (
    collect_examples,
    select_questions,
)
from mas_sae.experiments.collection_artifacts import save_collection_artifacts
from mas_sae.experiments.collection_config import load_collection_config
from mas_sae.experiments.provenance import (
    build_collection_progress_provenance,
    build_resolved_config,
)
from mas_sae.models.roles import resolve_roles, load_role_models, configure_agent
from mas_sae.activations.sites import resolve_solver_sites
from mas_sae.experiments.conditions import OMITTED


logger = logging.getLogger(__name__)
RESULT_ROOT = Path("results/collection")
ACTIVATION_ROOT = Path("data/activations")
DEFAULT_CHUNK_SIZE = 250

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a run whose chunks are under data/activations/<run>/_progress/<source_split>.",
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
    role_specs = resolve_roles(config)
    model_id = role_specs["solver"]["id"]
    source_split = config["dataset"]["source_split"]
    num_questions = config["dataset"]["num_questions"]
    layers = config["collection"]["layers"]
    seed = config["collection"]["seed"]
    protocol_version = config["collection"].get(
        "protocol_version", DEFAULT_PROTOCOL
    )
    run_name = config["output"]["run_name"]
    chunk_size = config["output"].get("chunk_size", DEFAULT_CHUNK_SIZE)
    progress = collection_progress.progress_dir(ACTIVATION_ROOT, run_name, source_split)

    if protocol_version not in PROTOCOLS:
        raise ValueError(
            "collection.protocol_version must be one of "
            f"{list(PROTOCOLS)}, got {protocol_version!r}."
        )
    protocol = PROTOCOLS[protocol_version]

    collection_progress.ensure_can_start(
        resume=args.resume,
        final_output_dir=RESULT_ROOT / run_name / source_split,
        progress=progress,
    )

    logger.info(
        "Starting run=%s split=%s questions=%d protocol=%s",
        run_name,
        source_split,
        num_questions,
        protocol_version,
    )
    logger.info("Loading model %s", model_id)

    loaded = load_role_models(role_specs)
    model, processor = loaded["solver"]
    candidate_sites = resolve_solver_sites(model, role_specs["solver"]["loader"], layers)
    solver = Solver(model, processor)
    critic = Critic(
        *loaded["critic"],
        blind_then_compare=protocol["blind_then_compare"],
        controlled_as_own_conclusion=protocol["controlled_as_own_conclusion"],
    )
    validator = Validator(*loaded["validator"])
    for role, agent in (("solver", solver), ("critic", critic), ("validator", validator)):
        configure_agent(agent, role_specs[role])

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

    manifest = selection["sampled_questions"]
    if manifest is None:
        manifest = describe_sampled_questions(selection["examples"], None)
    identity = build_collection_progress_provenance(
        config,
        resolved_config,
        manifest_sha256=collection_progress.manifest_sha256(manifest),
    )
    question_ids = [str(example["id"]) for example in selection["examples"]]

    if args.resume:
        collection_progress.check_identity(progress, identity)
        start_position = collection_progress.resume_position(progress, question_ids)
        logger.info("Resuming at question %d/%d", start_position + 1, len(question_ids))
    else:
        collection_progress.initialize(progress, identity)
        start_position = 0

    for start in range(start_position, len(question_ids), chunk_size):
        end = min(start + chunk_size, len(question_ids))
        logger.info("Collecting chunk %d-%d of %d", start + 1, end, len(question_ids))
        chunk = collect_examples(
            examples=selection["examples"][start:end],
            source_split=source_split,
            model=model,
            solver=solver,
            critic=critic,
            validator=validator,
            candidate_sites=candidate_sites,
            base_seed=seed,
            experiment_splits=selection["experiment_splits"],
            active_conditions=config["collection"].get("active_conditions", OMITTED),
            type_checked_target=protocol["type_checked_target"],
            question_offset=start,
        )
        collection_progress.write_chunk(
            progress,
            start=start,
            question_ids=question_ids[start:end],
            candidate_sites=candidate_sites,
            result=chunk,
        )

    result = collection_progress.merge_chunks(
        progress,
        question_ids=question_ids,
        candidate_sites=candidate_sites,
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
        exclusions=result["exclusions"],
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
