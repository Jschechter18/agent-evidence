"""Descriptive retention counts for paired Natural-critic arms.

Everything here is lexical; nothing is a semantic judge and nothing is an
inferential statistic. A row is an arm-eligible *disagreement* only when the
critic's blind answer and the Solver's Attempt 1 are both answer-like and
lexically different. Disagreement is not independence: the counts only say
what the critic advocated after seeing Attempt 1, with every denominator
stated so selection into each subset stays visible.
"""
from __future__ import annotations

from typing import Any

from mas_sae.evaluation.behavior import answer_kind, textual_relation

STAGE_FAILED = "stage_failed"
BLIND_KINDS = ("answer", "nonanswer", "uncertain", "missing", STAGE_FAILED)
BLIND_RELATIONS = ("same", "containment", "different", "solver_a1_not_answer_like")
OUTCOMES = ("retained_own_position", "moved_to_solver_a1", "third_position", "unclear")
RETAINED = "retained_own_position"


def candidate_status(row: dict[str, Any]) -> dict[str, Any]:
    """Classify one arm row on a lexical basis.

    ``blind_kind`` is the answer kind of the blind answer (``stage_failed``
    when the arm did not complete). ``blind_vs_solver_a1`` is set only for a
    committal (answer-like) blind answer. ``outcome`` is set only for an
    eligible disagreement: the advocated answer exactly matches the blind
    answer (retained), exactly matches Attempt 1 (moved), is a different
    answer (third position), or is anything else (unclear, including a
    noncommittal review or a containment match).
    """
    status = {"question_id": row.get("question_id"), "attempt1_sha256": row.get("attempt1_sha256"),
              "blind_kind": STAGE_FAILED,
              "blind_vs_solver_a1": None, "eligible": False, "outcome": None}
    if row.get("status") != "completed":
        return status

    blind, a1 = row.get("critic_blind_answer"), row.get("solver_attempt_1")
    status["blind_kind"] = answer_kind(blind)
    if status["blind_kind"] != "answer":
        return status
    if answer_kind(a1) != "answer":
        status["blind_vs_solver_a1"] = "solver_a1_not_answer_like"
        return status

    relation = textual_relation(blind, a1)
    status["blind_vs_solver_a1"] = "same" if relation == "exact" else relation
    if relation != "different":
        return status

    status["eligible"] = True
    advocated = row.get("critic_advocated_answer")
    if row.get("critic_noncommittal") is not False or answer_kind(advocated) != "answer":
        status["outcome"] = "unclear"
    elif textual_relation(advocated, blind) == "exact":
        status["outcome"] = RETAINED
    elif textual_relation(advocated, a1) == "exact":
        status["outcome"] = "moved_to_solver_a1"
    elif textual_relation(advocated, blind) == "different" and textual_relation(advocated, a1) == "different":
        status["outcome"] = "third_position"
    else:
        status["outcome"] = "unclear"
    return status


def _counts(values: list[Any], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: sum(value == key for value in values) for key in keys}


def arm_summary(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    """Blind-answer and disagreement-outcome counts for one arm, with denominators."""
    committal = [status for status in statuses if status["blind_kind"] == "answer"]
    eligible = [status for status in statuses if status["eligible"]]
    return {
        "blind_kind": {"denominator": "all arm rows", "n": len(statuses),
                       "counts": _counts([s["blind_kind"] for s in statuses], BLIND_KINDS)},
        "blind_vs_solver_a1": {"denominator": "rows with a committal (answer-like) blind answer",
                               "n": len(committal),
                               "counts": _counts([s["blind_vs_solver_a1"] for s in committal], BLIND_RELATIONS)},
        "disagreement_outcomes": {"denominator": "arm-eligible disagreements (blind and Solver A1 both "
                                                 "answer-like and lexically different)",
                                  "n": len(eligible),
                                  "counts": _counts([s["outcome"] for s in eligible], OUTCOMES)},
    }


def _index_arm(statuses: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for status in statuses:
        question_id = status.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"{name}: question_id must be present and non-empty.")
        if question_id in indexed:
            raise ValueError(f"{name}: duplicate question_id {question_id!r}.")
        digest = status.get("attempt1_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{name}: {question_id!r} requires a valid attempt1_sha256.")
        indexed[question_id] = status
    return indexed


def paired_retention(statuses_a: list[dict[str, Any]], statuses_b: list[dict[str, Any]],
                     *, name_a: str, name_b: str) -> dict[str, Any]:
    """Descriptive 2x2 on jointly eligible, classifiable rows sharing saved A1."""
    by_a, by_b = _index_arm(statuses_a, name_a), _index_arm(statuses_b, name_b)
    for question_id in by_a.keys() & by_b.keys():
        if by_a[question_id]["attempt1_sha256"] != by_b[question_id]["attempt1_sha256"]:
            raise ValueError(f"{question_id!r}: attempt1_sha256 differs across arms.")
    pairs = [(status, by_b[status["question_id"]]) for status in statuses_a
             if status["eligible"] and status["question_id"] in by_b and by_b[status["question_id"]]["eligible"]]
    classifiable = {RETAINED, "moved_to_solver_a1", "third_position"}
    binary_pairs = [(a, b) for a, b in pairs
                    if a["outcome"] in classifiable and b["outcome"] in classifiable]
    n_unclear = sum(a["outcome"] == "unclear" or b["outcome"] == "unclear" for a, b in pairs)
    cells = {"both_retained": 0, f"{name_a}_only_retained": 0, f"{name_b}_only_retained": 0,
             "neither_retained": 0}
    for status_a, status_b in binary_pairs:
        a, b = status_a["outcome"] == RETAINED, status_b["outcome"] == RETAINED
        key = ("both_retained" if a and b else f"{name_a}_only_retained" if a
               else f"{name_b}_only_retained" if b else "neither_retained")
        cells[key] += 1
    return {"denominator": "questions eligible in both arms with classifiable binary retention in both arms",
            "n_jointly_eligible": len(pairs),
            "n_jointly_classifiable_for_binary_retention": len(binary_pairs),
            "n_jointly_eligible_excluded_unclear": n_unclear,
            "n_eligible_by_arm": {name_a: sum(s["eligible"] for s in statuses_a),
                                  name_b: sum(s["eligible"] for s in statuses_b)},
            "counts": cells}


def retention_report(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per-arm counts plus a secondary paired 2x2 for exactly two arms."""
    if len(rows_by_arm) != 2:
        raise ValueError("retention_report compares exactly two arms")
    statuses = {arm: [candidate_status(row) for row in rows] for arm, rows in rows_by_arm.items()}
    (name_a, statuses_a), (name_b, statuses_b) = statuses.items()
    return {
        "arms": {arm: arm_summary(arm_statuses) for arm, arm_statuses in statuses.items()},
        "paired": paired_retention(statuses_a, statuses_b, name_a=name_a, name_b=name_b),
        "matching_basis": "normalized_text_not_semantic",
    }
