from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from mas_sae.activations.capture import MultiSiteCapture
from mas_sae.agents.critic import Critic, CriticCondition, CriticBlindAnswerError
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.evaluation.scoring import (
    answer_matches,
    interaction_labels,
)
from mas_sae.experiments.controlled_targets import (
    ControlledTarget,
    DistractorRequest,
    choose_controlled_incorrect_target,
)
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.experiments.conditions import OMITTED, resolve_conditions
from mas_sae.evaluation.behavior import classify_behavior


def choose_incorrect_answer(
    *,
    paragraphs: list[dict[str, Any]],
    gold: str,
    aliases: Iterable[str] = (),
) -> str:
    """Choose a deterministic wrong answer for controlled feedback."""

    aliases = tuple(aliases)

    # Prefer a title from a non-supporting paragraph.
    candidates = [
        paragraph
        for paragraph in paragraphs
        if not paragraph.get("is_supporting", False)
    ]

    candidates.extend(
        paragraph
        for paragraph in paragraphs
        if paragraph.get("is_supporting", False)
    )

    for paragraph in candidates:
        candidate = str(
            paragraph.get("title", "")
        ).strip()

        if (
            candidate
            and not answer_matches(
                candidate,
                gold,
                aliases,
            )
        ):
            return candidate

    fallback = f"Not {gold}"

    if answer_matches(
        fallback,
        gold,
        aliases,
    ):
        raise ValueError(
            "Could not construct a controlled incorrect answer."
        )

    return fallback


def choose_type_checked_incorrect_target(
    *,
    question_id: str,
    question: str,
    paragraphs: list[dict[str, Any]],
    gold: str,
    aliases: tuple[str, ...],
    attempt1: str,
    decomposition: Sequence[dict[str, Any]],
    critic: Critic,
) -> ControlledTarget:
    """Type-checked wrong target: hop answer, context span, LLM distractor,
    or raise. The critic supplies the LLM distractor step."""

    def generate_distractor(request: DistractorRequest) -> str:
        return critic.propose_distractor(
            question=request.question,
            paragraphs=request.paragraphs,
            type_description=request.type_description,
            excluded=request.excluded,
        )

    return choose_controlled_incorrect_target(
        question_id=question_id,
        question=question,
        paragraphs=paragraphs,
        gold=gold,
        aliases=aliases,
        attempt1=attempt1,
        decomposition=decomposition,
        distractor_generator=generate_distractor,
    )


def _controlled_target_fields(
    condition: CriticCondition,
    controlled_target: ControlledTarget | None,
) -> dict[str, Any]:
    """Target-provenance record fields for one episode (null where not
    applicable). Only used when the type-checked target generator ran."""
    if condition is CriticCondition.NATURAL:
        return {
            "controlled_target_source": None,
            "controlled_target_type_check": None,
        }

    if condition is CriticCondition.CONTROLLED_CORRECT:
        return {
            "controlled_target_source": "gold",
            "controlled_target_type_check": None,
        }

    if controlled_target is None:
        raise ValueError(
            "controlled_target is required for the controlled-incorrect "
            "condition."
        )

    return {
        "controlled_target_source": controlled_target.source.value,
        "controlled_target_type_check": controlled_target.type_check,
    }


def run_question(
    *,
    question_id: str,
    question: str,
    paragraphs: list[dict[str, Any]],
    gold: str,
    aliases: Iterable[str],
    model: Any,
    solver: Solver,
    critic: Critic,
    validator: Validator,
    candidate_sites: list[str],
    seed: int,
    decomposition: Sequence[dict[str, Any]] = (),
    type_checked_target: bool = False,
    active_conditions=OMITTED,
) -> dict[str, Any]:
    """
    Run one complete paired Solver-Critic experiment.

    Solver Attempt 1 is generated exactly once and reused across
    enabled feedback conditions (all three by default for V1).

    Two optional capabilities extend the records; with both off the
    original record schema is reproduced exactly.

    - When ``critic.blind_then_compare`` is set, the critic answers the
      question blind before any prompt containing Attempt 1, and the
      natural episode records ``critic_blind_answer``. A missing blind
      answer raises ``CriticBlindAnswerError``.
    - When ``type_checked_target`` is set, the controlled-incorrect
      target comes from the type-checked generator (hop answer, context
      span, LLM distractor via the critic, or ``ControlledTargetError``)
      using ``decomposition``, and every episode records
      ``controlled_target_source`` and ``controlled_target_type_check``.
      Otherwise the paragraph-title heuristic is used.

    Either error aborts this question; ``collect_examples`` skips it and
    continues with the next one.
    """

    conditions = resolve_conditions(active_conditions)
    aliases = tuple(aliases)

    seed_everything(seed)

    # ---------------------------------------------------------
    # Attempt 1: generate exactly once.
    # ---------------------------------------------------------

    with MultiSiteCapture(
        model,
        candidate_sites,
    ) as capture:
        attempt1 = solver.solve(
            question,
            paragraphs,
        )

    attempt1_activations = capture.activations
    a1_telemetry = getattr(solver, "last_generation", None)

    attempt1_correct = answer_matches(
        attempt1,
        gold,
        aliases,
    )

    # ---------------------------------------------------------
    # Blind-then-compare: the critic's blind answer is formed once
    # per question, before any prompt that contains Attempt 1.
    # ---------------------------------------------------------

    blind_telemetry = None
    blind_raw_output = None
    blind_answer: str | None = None
    controlled_target: ControlledTarget | None = None

    if critic.blind_then_compare and CriticCondition.NATURAL in conditions:
        try:
            blind_answer = critic.answer_blind(question, paragraphs)
        except CriticBlindAnswerError as error:
            error.collection_context = {"stage": "critic_blind", "a1_exists": True,
                                        "critic_exists": True, "a2_exists": False}
            raise
        blind_telemetry = getattr(critic, "last_generation", None)
        blind_raw_output = critic.last_blind_raw_output

    if type_checked_target and CriticCondition.CONTROLLED_INCORRECT in conditions:
        controlled_target = choose_type_checked_incorrect_target(
            question_id=question_id,
            question=question,
            paragraphs=paragraphs,
            gold=gold,
            aliases=aliases,
            attempt1=attempt1,
            decomposition=decomposition,
            critic=critic,
        )
        incorrect_answer = controlled_target.answer
    elif CriticCondition.CONTROLLED_INCORRECT in conditions:
        incorrect_answer = choose_incorrect_answer(
            paragraphs=paragraphs,
            gold=gold,
            aliases=aliases,
        )

    episodes: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # Enabled conditions branch from SAME Attempt 1.
    # ---------------------------------------------------------

    for condition in conditions:
        if condition is CriticCondition.NATURAL:
            target_answer = None

        elif condition is CriticCondition.CONTROLLED_CORRECT:
            target_answer = gold

        else:
            target_answer = incorrect_answer

        telemetry_start = len(getattr(critic, "generation_history", []))
        feedback = critic.critique(
            question=question,
            paragraphs=paragraphs,
            solver_answer=attempt1,
            condition=condition,
            advocated_answer=target_answer,
            blind_answer=(
                blind_answer
                if condition is CriticCondition.NATURAL
                else None
            ),
        )

        feedback_text = feedback.to_solver_text()

        # -----------------------------------------------------
        # Attempt 2 after Critic feedback.
        # -----------------------------------------------------

        with MultiSiteCapture(
            model,
            candidate_sites,
        ) as capture:
            attempt2 = solver.revise(
                question,
                paragraphs,
                attempt1,
                feedback_text,
            )

        attempt2_activations = capture.activations

        labels = interaction_labels(
            attempt1=attempt1,
            attempt2=attempt2,
            advocated_answer=feedback.advocated_answer,
            gold=gold,
            aliases=aliases,
        )

        # Validator judges final system output.
        validator_verdict = validator.validate(
            question,
            gold,
            aliases,
            attempt2,
        )

        record = {
            "episode_id": (
                f"{question_id}__{condition.value}"
            ),
            "question_id": question_id,
            "question": question,
            "ground_truth": gold,
            "answer_aliases": list(aliases),

            "solver_attempt_1": attempt1,
            "solver_attempt_1_correct": attempt1_correct,

            "critic_condition": condition.value,
            "critic_verdict": feedback.verdict,
            "critic_advocated_answer": (
                feedback.advocated_answer
            ),
            "critic_feedback": feedback_text,
            "critic_raw_output": feedback.raw_output,
            "critic_noncommittal": feedback.noncommittal,

            "solver_attempt_2": attempt2,

            **labels,

            "validator_final_correct": (
                validator_verdict.is_correct
            ),
            "validator_raw_output": (
                validator_verdict.raw_output
            ),

            "seed": seed,
        }

        if active_conditions is not OMITTED:
            record.update(classify_behavior(
                attempt1, attempt2, feedback.advocated_answer,
                usable=not feedback.noncommittal,
            ))
            record["generation_telemetry"] = {
                "critic_review": getattr(critic, "generation_history", [])[telemetry_start:],
                "solver_a1": a1_telemetry,
                "solver_a2": getattr(solver, "last_generation", None),
                "critic_blind": blind_telemetry,
                "validator": getattr(validator, "last_generation", None),
            }

        if critic.blind_then_compare:
            record["critic_blind_answer"] = feedback.blind_answer
            if condition is CriticCondition.NATURAL:
                record["critic_blind_raw_output"] = blind_raw_output

        if type_checked_target:
            record.update(
                _controlled_target_fields(condition, controlled_target)
            )

        episodes.append(
            {
                "record": record,
                "attempt2_activations": (
                    attempt2_activations
                ),
            }
        )

    return {
        "attempt1": attempt1,
        "attempt1_activations": (
            attempt1_activations
        ),
        "episodes": episodes,
    }
