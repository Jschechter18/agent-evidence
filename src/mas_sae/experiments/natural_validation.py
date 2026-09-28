"""Behavioural validation of the Natural blind-first critic on paired arms.

One Solver generates Attempt 1 exactly once per question; that saved
Attempt 1 is then reviewed by two critic arms (``same_model``
and ``cross_model``), each of which answers blind first, compares, and
sends feedback to the Solver for a separate Attempt 2. Nothing here
captures activations, runs a Validator, or executes controlled conditions.

The questions come from a frozen manifest (ids only) whose bytes are hash
checked; examples are reloaded from the pinned dataset revision, so
contexts are reconstructed rather than copied from historical rows.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from mas_sae.agents.critic import (
    NATURAL_BLIND_PROMPT,
    NATURAL_COMPARE_PROMPT,
    CriticBlindAnswerError,
    CriticCondition,
)
from mas_sae.agents.solver import Solver
from mas_sae.data.musique import (
    MUSIQUE_DATASET_ID,
    SUPPORTED_SOURCE_SPLITS,
    hop_group,
    hop_type,
    load_musique_examples_by_id,
)
from mas_sae.evaluation.behavior import classify_behavior, textual_relation
from mas_sae.evaluation.critic_retention import retention_report
from mas_sae.evaluation.scoring import answer_matches
from mas_sae.experiments import artifacts
from mas_sae.experiments.artifacts import sha256_file, sha256_json, sha256_text
from mas_sae.experiments.provenance import environment_metadata, role_metadata
from mas_sae.experiments.records import append_jsonl, read_jsonl
from mas_sae.experiments.reproducibility import (
    seed_everything,
    validate_commit_revision,
)
from mas_sae.models.roles import resolve_role_spec, weight_identity

logger = logging.getLogger(__name__)

PROTOCOL = "natural_blind_first"
SOLVER_ROLE = "solver"
ARM_ROLES = {"same_model": "same_model_critic", "cross_model": "cross_model_critic"}
DEFAULT_TOKENS = {"solver": 32, "same_model_critic": 256, "cross_model_critic": 256}
ATTEMPT1_FILE = "attempt1.jsonl"
PROVENANCE_FILE = "provenance.json"
SUMMARY_FILE = "summary.json"
PROMPT_VERSIONS = {
    "solver_solve": ("SOLVE_PROMPT_V1", Solver.SOLVE_PROMPT_V1),
    "solver_revise": ("REVISE_PROMPT_V1", Solver.REVISE_PROMPT_V1),
    "critic_blind": ("NATURAL_BLIND_PROMPT", NATURAL_BLIND_PROMPT),
    "critic_compare": ("NATURAL_COMPARE_PROMPT", NATURAL_COMPARE_PROMPT),
}


def arm_file(arm: str) -> str:
    return f"arm_{arm}.jsonl"


# ---------------------------------------------------------------------------
# Configuration and frozen inputs
# ---------------------------------------------------------------------------

def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping.")
    return value


def _require_text(section: dict[str, Any], key: str, name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string.")
    return value


def load_validation_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one Natural-validation YAML config.

    The config pins the dataset revision, the manifest bytes, three model
    roles (``solver``, ``same_model_critic``, ``cross_model_critic``), the
    protocol and the seed. The same-model critic must load exactly the
    Solver's weights (same id, revision, loader, dtype and placement).
    """
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("Validation config must be a mapping.")
    missing = {"dataset", "manifest", "roles", "validation", "output"} - raw.keys()
    if missing:
        raise ValueError(f"Validation config is missing sections: {sorted(missing)}.")

    dataset = _require_mapping(raw, "dataset")
    if _require_text(dataset, "repo", "dataset") != MUSIQUE_DATASET_ID:
        raise ValueError(f"dataset.repo must be {MUSIQUE_DATASET_ID!r}; the loader supports no other dataset.")
    validate_commit_revision(dataset.get("revision"), "dataset")
    if dataset.get("source_split") not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(f"dataset.source_split must be one of {sorted(SUPPORTED_SOURCE_SPLITS)}.")

    manifest = _require_mapping(raw, "manifest")
    _require_text(manifest, "path", "manifest")
    digest = _require_text(manifest, "sha256", "manifest")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("manifest.sha256 must be a lowercase 64-character hex digest.")

    roles = _require_mapping(raw, "roles")
    if set(roles) != set(DEFAULT_TOKENS):
        raise ValueError(f"roles must specify exactly {sorted(DEFAULT_TOKENS)}.")
    resolved = {role: resolve_role_spec(role, roles[role], default_max_new_tokens=DEFAULT_TOKENS[role])
                for role in DEFAULT_TOKENS}
    for role, spec in resolved.items():
        validate_commit_revision(spec["revision"], f"roles.{role}")
    if weight_identity(resolved["same_model_critic"]) != weight_identity(resolved[SOLVER_ROLE]):
        raise ValueError("roles.same_model_critic must load exactly the Solver's checkpoint "
                         "(same id, revision, loader, dtype, placement and cache_dir).")

    validation = _require_mapping(raw, "validation")
    if validation.get("protocol") != PROTOCOL:
        raise ValueError(f"validation.protocol must be {PROTOCOL!r}.")
    if type(validation.get("seed")) is not int:
        raise ValueError("validation.seed must be an integer.")

    output = _require_mapping(raw, "output")
    _require_text(output, "run_name", "output")
    return {**raw, "roles": resolved}


def load_manifest(path: str | Path, expected_sha256: str) -> tuple[list[dict[str, Any]], str]:
    """Read the frozen question manifest after checking its exact bytes."""
    manifest_path = Path(path)
    actual = sha256_file(manifest_path)
    if actual != expected_sha256:
        raise ValueError(f"Manifest {manifest_path} has sha256 {actual}, expected {expected_sha256}.")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries or not all(isinstance(e, dict) for e in entries):
        raise ValueError("Manifest must be a non-empty list of objects.")
    ids = [str(entry.get("question_id", "")) for entry in entries]
    if any(not question_id for question_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Manifest question_id values must be present and unique.")
    positions = [entry.get("position") for entry in entries]
    if positions != list(range(len(entries))):
        raise ValueError("Manifest positions must be 0..n-1 in file order.")
    return entries, actual


def load_validation_examples(config: dict[str, Any], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reload the manifest's questions from the pinned dataset revision, in manifest order."""
    dataset = config["dataset"]
    return load_musique_examples_by_id(
        dataset["source_split"],
        [entry["question_id"] for entry in manifest],
        revision=dataset["revision"],
    )


def input_digests(example: dict[str, Any]) -> dict[str, Any]:
    """Hashes of the exact question and context the Solver will see."""
    paragraphs = example["paragraphs"]
    return {
        "input_sha256": sha256_json({"question_id": str(example["id"]), "question": example["question"],
                                     "paragraphs": paragraphs}),
        "context_sha256": sha256_text(Solver.format_paragraphs(paragraphs)),
        "num_paragraphs": len(paragraphs),
    }


def inputs_sha256(examples: list[dict[str, Any]]) -> str:
    """One digest over all per-question input digests, in order."""
    return sha256_json([input_digests(example)["input_sha256"] for example in examples])


def config_sha256(config: dict[str, Any]) -> str:
    """Digest of every scientifically meaningful setting (all but ``output``).

    Covers the resolved roles (id, revision, dtype, placement, generation,
    chat-template options), dataset revision and split, manifest, protocol
    and seed; a resumed run must match it exactly.
    """
    return sha256_json({key: value for key, value in config.items() if key != "output"})


def _telemetry(agent: Any) -> dict[str, Any]:
    generation = getattr(agent, "last_generation", None)
    prompt = getattr(agent, "last_prompt", None)
    return {"generation": dict(generation) if isinstance(generation, dict) else None,
            "prompt": dict(prompt) if isinstance(prompt, dict) else None}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def build_validation_provenance(config: dict[str, Any], agents: dict[str, Any], *,
                                manifest_sha256: str, inputs_digest: str,
                                num_questions: int) -> dict[str, Any]:
    """Everything needed to rerun the validation, independent of the run name."""
    dataset = config["dataset"]
    return {
        "created_at_utc": artifacts.utc_now().isoformat(),
        "git_commit": artifacts.get_git_commit(),
        "command": artifacts.get_run_command(),
        "protocol": PROTOCOL,
        "dataset": {"dataset_id": dataset["repo"], "requested_revision": dataset["revision"],
                    "source_split": dataset["source_split"]},
        "manifest": {"path": config["manifest"]["path"], "sha256": manifest_sha256,
                     "num_questions": num_questions},
        "config_sha256": config_sha256(config),
        "inputs_sha256": inputs_digest,
        "hash_basis": {
            "config_sha256": "sha256 of the canonical JSON resolved config without its output section",
            "input_sha256": "sha256 of canonical JSON {question_id, question, paragraphs}",
            "context_sha256": "sha256 of Solver.format_paragraphs(paragraphs)",
            "inputs_sha256": "sha256 of the canonical JSON list of input_sha256 in manifest order",
            "attempt1_sha256": "sha256 of the saved Solver Attempt 1 text",
            "prompt.input_ids_sha256": "sha256 of the exact tokenized prompt ids of each generation",
        },
        "roles": {role: role_metadata(agent) for role, agent in agents.items()},
        "arms": dict(ARM_ROLES),
        "seed": config["validation"]["seed"],
        "seed_policy": (
            "validation.seed + manifest position, reset before each newly generated Attempt 1 "
            "and before each newly executed critic arm/question sequence; no reset between "
            "blind answer, compare, and Solver revision; reused rows do not reset the seed"),
        "prompt_versions": {key: name for key, (name, _) in PROMPT_VERSIONS.items()},
        "prompt_template_sha256": {key: sha256_text(text) for key, (_, text) in PROMPT_VERSIONS.items()},
        "capabilities": {"blind_then_compare": True, "activation_capture": False,
                         "validator": False, "controlled_conditions": False},
        "environment": environment_metadata(),
        "package_versions": {name: artifacts.get_package_version(name)
                             for name in ("torch", "transformers", "datasets", "huggingface_hub")},
        "historical_context_note": (
            "Questions are the historical manifest ids; contexts are reconstructed from the pinned "
            "dataset revision by the deterministic loader. Exact byte identity to the earlier diagnostic "
            "cannot be independently verified because historical context hashes were not recorded."),
    }


# ---------------------------------------------------------------------------
# Stage 1: Attempt 1, generated once and persisted before any critic runs
# ---------------------------------------------------------------------------

def _read_existing_rows(path: Path, examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate all persisted question IDs before any rows are reused or generated."""
    expected_ids = {str(example["id"]) for example in examples}
    existing = {}
    for row in read_jsonl(path) if path.exists() else []:
        question_id = row.get("question_id") if isinstance(row, dict) else None
        if not isinstance(question_id, str) or not question_id.strip():
            raise RuntimeError(f"{path}: persisted row requires a non-empty question_id.")
        if question_id in existing:
            raise RuntimeError(f"{path}: duplicate persisted question_id {question_id!r}.")
        if question_id not in expected_ids:
            raise RuntimeError(f"{path}: persisted question_id {question_id!r} is outside the manifest.")
        existing[question_id] = row
    return existing


def generate_attempt1(*, solver: Any, examples: list[dict[str, Any]], base_seed: int,
                      path: str | Path) -> list[dict[str, Any]]:
    """Generate and durably persist one Solver Attempt 1 per question.

    Rows already present in ``path`` are reused after checking that they
    belong to the same question, seed and input hashes; they are never
    regenerated. Any persisted row that the manifest does not contain is an
    error, so a stale file cannot be silently mixed into a run.
    """
    path = Path(path)
    existing = _read_existing_rows(path, examples)
    rows: list[dict[str, Any]] = []
    for position, example in enumerate(examples):
        question_id = str(example["id"])
        expected = {"question_id": question_id, "position": position, "seed": base_seed + position,
                    **input_digests(example)}
        if question_id in existing:
            row = existing[question_id]
            mismatch = sorted(key for key, value in expected.items() if row.get(key) != value)
            if mismatch or not isinstance(row.get("solver_attempt_1"), str) \
                    or row.get("attempt1_sha256") != sha256_text(row["solver_attempt_1"]):
                raise RuntimeError(f"Persisted Attempt 1 for {question_id} does not match the current "
                                   f"inputs (mismatch: {mismatch}); refusing to regenerate silently.")
            rows.append(row)
            continue
        seed_everything(expected["seed"])
        attempt1 = solver.solve(example["question"], example["paragraphs"])
        row = {**expected, "solver_attempt_1": attempt1, "attempt1_sha256": sha256_text(attempt1),
               "solver_generation": _telemetry(solver)}
        append_jsonl(path, row)
        rows.append(row)
        logger.info("Attempt 1 %d/%d (%s)", position + 1, len(examples), question_id)
    return rows


# ---------------------------------------------------------------------------
# Stage 2: one critic arm over the saved Attempt 1
# ---------------------------------------------------------------------------

def _gold_labels(example: dict[str, Any], *, blind: str | None, advocated: str | None,
                 attempt1: str, attempt2: str | None) -> dict[str, Any]:
    """Post-hoc correctness labels; computed after generation, never shown to a critic."""
    gold, aliases = example["answer"], list(example.get("answer_aliases") or [])
    return {
        "ground_truth": gold, "answer_aliases": aliases,
        "solver_attempt_1_correct": answer_matches(attempt1, gold, aliases),
        "critic_blind_correct": None if blind is None else answer_matches(blind, gold, aliases),
        "critic_feedback_correct": None if advocated is None else answer_matches(advocated, gold, aliases),
        "solver_attempt_2_correct": None if attempt2 is None else answer_matches(attempt2, gold, aliases),
    }


def run_critic_arm(*, arm: str, critic: Any, solver: Any, examples: list[dict[str, Any]],
                   attempt1_rows: list[dict[str, Any]], base_seed: int, path: str | Path) -> list[dict[str, Any]]:
    """Blind answer, compare review, and Solver revision for one critic arm.

    Every question uses the saved Attempt 1 from ``attempt1_rows`` (hash
    checked). A failed blind step is recorded as an explicit failed row and
    the question continues in the other arm untouched. Rows already in
    ``path`` are reused, not rerun.
    """
    if arm not in ARM_ROLES:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {sorted(ARM_ROLES)}.")
    if not getattr(critic, "blind_then_compare", False):
        raise ValueError(f"The {arm} critic must be configured with blind_then_compare=True.")
    path = Path(path)
    attempt1_by_id = {row["question_id"]: row for row in attempt1_rows}
    existing = _read_existing_rows(path, examples)
    rows: list[dict[str, Any]] = []
    for position, example in enumerate(examples):
        question_id = str(example["id"])
        if question_id not in attempt1_by_id:
            raise RuntimeError(f"No saved Attempt 1 for {question_id}; run generate_attempt1 first.")
        saved = attempt1_by_id[question_id]
        attempt1 = saved["solver_attempt_1"]
        if saved["attempt1_sha256"] != sha256_text(attempt1):
            raise RuntimeError(f"Saved Attempt 1 for {question_id} fails its hash check.")
        base = {"arm": arm, "protocol": PROTOCOL, "question_id": question_id, "position": position,
                "seed": base_seed + position, "hop_type": hop_type(question_id),
                "hop_group": hop_group(question_id), "solver_attempt_1": attempt1,
                "attempt1_sha256": saved["attempt1_sha256"], "input_sha256": saved["input_sha256"],
                "context_sha256": saved["context_sha256"]}
        if question_id in existing:
            row = existing[question_id]
            mismatch = sorted(key for key, value in base.items() if row.get(key) != value)
            if mismatch or not isinstance(row.get("solver_attempt_1"), str) \
                    or row.get("attempt1_sha256") != sha256_text(row["solver_attempt_1"]):
                raise RuntimeError(f"Persisted {arm} row for {question_id} belongs to different inputs.")
            rows.append(row)
            continue
        seed_everything(base["seed"])
        try:
            blind = critic.answer_blind(example["question"], example["paragraphs"])
        except CriticBlindAnswerError as error:
            row = {**base, "status": "failed", "failed_stage": "blind",
                   "error_type": type(error).__name__, "reason": str(error),
                   "critic_blind_answer": None, "critic_blind_raw_output": critic.last_blind_raw_output,
                   "critic_blind_generation": _telemetry(critic),
                   **_gold_labels(example, blind=None, advocated=None, attempt1=attempt1, attempt2=None)}
            append_jsonl(path, row)
            rows.append(row)
            logger.warning("%s arm: blind step failed for %s: %s", arm, question_id, error)
            continue
        blind_telemetry = _telemetry(critic)
        feedback = critic.critique(question=example["question"], paragraphs=example["paragraphs"],
                                   solver_answer=attempt1, condition=CriticCondition.NATURAL,
                                   blind_answer=blind)
        review_telemetry = _telemetry(critic)
        feedback_text = feedback.to_solver_text()
        attempt2 = solver.revise(example["question"], example["paragraphs"], attempt1, feedback_text)
        advocated = feedback.advocated_answer
        row = {**base, "status": "completed",
               "critic_blind_answer": blind, "critic_blind_raw_output": critic.last_blind_raw_output,
               "critic_verdict": feedback.verdict, "critic_advocated_answer": advocated,
               "critic_explanation": feedback.explanation, "critic_raw_output": feedback.raw_output,
               "critic_noncommittal": feedback.noncommittal, "critic_feedback_text": feedback_text,
               "solver_attempt_2": attempt2,
               "blind_vs_solver": textual_relation(blind, attempt1),
               "advocated_vs_blind": textual_relation(advocated, blind),
               **classify_behavior(attempt1, attempt2, advocated, usable=not feedback.noncommittal),
               **_gold_labels(example, blind=blind, advocated=advocated, attempt1=attempt1, attempt2=attempt2),
               "critic_blind_generation": blind_telemetry, "critic_review_generation": review_telemetry,
               "solver_revise_generation": _telemetry(solver)}
        append_jsonl(path, row)
        rows.append(row)
        logger.info("%s arm %d/%d (%s)", arm, position + 1, len(examples), question_id)
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_summary(attempt1_rows: list[dict[str, Any]], rows_by_arm: dict[str, list[dict[str, Any]]],
                  examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive counts only, each with its denominator; no inferential statistics.

    Critic position (``critic_position_relation``) and Solver behaviour
    (``solver_behavior``) stay separate labels: a changed Solver answer is
    not counted as accepting the critic. Gold correctness is post-hoc.
    """
    retention = retention_report(rows_by_arm)
    arms = {}
    for arm, rows in rows_by_arm.items():
        completed = [row for row in rows if row.get("status") == "completed"]
        arms[arm] = {
            "n_rows": len(rows), "n_completed": len(completed),
            "n_failed": len(rows) - len(completed),
            "failed_stage_counts": _count([row for row in rows if row.get("status") != "completed"], "failed_stage"),
            **retention["arms"][arm],
            "completed_denominator": "completed arm rows",
            "critic_position_relation_counts": _count(completed, "critic_position_relation"),
            "solver_behavior_counts": _count(completed, "solver_behavior"),
            "n_solver_attempt_2_correct_post_hoc": sum(row.get("solver_attempt_2_correct") is True
                                                       for row in completed),
        }
    return {"protocol": PROTOCOL, "num_questions": len(attempt1_rows),
            "n_solver_attempt_1_correct_post_hoc": sum(
                answer_matches(row["solver_attempt_1"], example["answer"], example.get("answer_aliases") or [])
                for row, example in zip(attempt1_rows, examples)),
            "arms": arms, "paired_secondary": retention["paired"],
            "matching_basis": retention["matching_basis"]}


def run_natural_validation(*, config: dict[str, Any], output_dir: str | Path, solver: Any,
                           critics: dict[str, Any], agents: dict[str, Any], examples: list[dict[str, Any]],
                           manifest: list[dict[str, Any]], manifest_sha256: str,
                           resume: bool = False) -> dict[str, Any]:
    """Run Attempt 1 once, then both critic arms, writing everything under ``output_dir``.

    ``critics`` maps arm name to a blind-then-compare ``Critic``; ``agents``
    maps role name to the configured agent for provenance. With ``resume``
    an existing output directory is continued (completed rows are reused);
    otherwise an existing directory is an error.
    """
    if set(critics) != set(ARM_ROLES):
        raise ValueError(f"critics must map exactly {sorted(ARM_ROLES)}.")
    if [str(example["id"]) for example in examples] != [entry["question_id"] for entry in manifest]:
        raise ValueError("examples must be the manifest questions in manifest order.")
    output_dir = Path(output_dir)
    if not resume:
        artifacts.ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_digest = inputs_sha256(examples)
    provenance_path = output_dir / PROVENANCE_FILE
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("manifest", {}).get("sha256") != manifest_sha256 \
                or previous.get("inputs_sha256") != inputs_digest \
                or previous.get("config_sha256") != config_sha256(config) \
                or previous.get("prompt_template_sha256") != {
                    key: sha256_text(text) for key, (_, text) in PROMPT_VERSIONS.items()}:
            raise RuntimeError(f"{provenance_path} was written for a different configuration, inputs, or prompt templates; "
                               "refusing to resume.")
    else:
        if resume and any((output_dir / name).exists()
                          for name in (ATTEMPT1_FILE, *(arm_file(arm) for arm in ARM_ROLES))):
            raise RuntimeError(f"{provenance_path} is missing but persisted JSONL files exist; "
                               "refusing to resume without provenance.")
        provenance = build_validation_provenance(config, agents, manifest_sha256=manifest_sha256,
                                                 inputs_digest=inputs_digest, num_questions=len(examples))
        provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Check every persisted stage before an earlier stage can append new rows.
    for name in (ATTEMPT1_FILE, *(arm_file(arm) for arm in ARM_ROLES)):
        _read_existing_rows(output_dir / name, examples)

    base_seed = config["validation"]["seed"]
    attempt1_rows = generate_attempt1(solver=solver, examples=examples, base_seed=base_seed,
                                      path=output_dir / ATTEMPT1_FILE)
    rows_by_arm = {arm: run_critic_arm(arm=arm, critic=critics[arm], solver=solver, examples=examples,
                                       attempt1_rows=attempt1_rows, base_seed=base_seed,
                                       path=output_dir / arm_file(arm))
                   for arm in ARM_ROLES}
    summary = build_summary(attempt1_rows, rows_by_arm, examples)
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return summary
