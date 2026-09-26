"""Lexical behavior classification; no semantic equivalence is claimed."""
import pytest

from mas_sae.evaluation.behavior import classify_behavior


@pytest.mark.parametrize("a1,a2,adv,relation,outcome", [
    ("Paris", "Paris", "Paris", "same_position", "no_conflict"),
    ("Paris", "Paris", "London", "different_nonrefusal_candidate", "retained_solver_a1"),
    ("Paris", "London", "London", "different_nonrefusal_candidate", "adopted_critic"),
    ("Paris", "Rome", "London", "different_nonrefusal_candidate", "third_answer_revision"),
    ("Paris", "Rome", None, "ambiguous_or_unresolved", "ambiguous"),
    ("Paris", "Rome", "I cannot determine the answer", "refusal_or_nonanswer", "ambiguous"),
    ("Paris", "Rome", "N/A", "refusal_or_nonanswer", "ambiguous"),
    ("Paris", "Rome", "Paris, France", "ambiguous_or_unresolved", "ambiguous"),
    ("Paris", "Maybe London", "London", "different_nonrefusal_candidate", "ambiguous"),
])
def test_behavior(a1, a2, adv, relation, outcome):
    labels = classify_behavior(a1, a2, adv)
    assert labels["critic_position_relation"] == relation
    assert labels["solver_behavior"] == outcome
    if outcome in {"ambiguous", "no_conflict"}:
        assert labels["direct_critic_adoption"] is None


def test_malformed_feedback_stays_ambiguous():
    assert classify_behavior("Paris", "London", "London", usable=False)["solver_behavior"] == "ambiguous"


@pytest.mark.parametrize("refusal", [
    "I do not have enough information to answer.",
    "I'm sorry, I can't answer that.",
    "I’m sorry, I can’t answer that.",
    "The context does not identify the person.",
    "There is no evidence in the passage.",
])
def test_refusal_copying_is_not_candidate_adoption(refusal):
    labels = classify_behavior("Paris", refusal, refusal)
    assert labels["critic_position_relation"] == "refusal_or_nonanswer"
    assert labels["solver_behavior"] == "ambiguous"
    assert labels["direct_critic_adoption"] is None
    assert labels["solver_copied_nonanswer"] is True
    assert labels["critic_textual_relation"] == "different"


def test_lexical_difference_does_not_claim_semantic_disagreement():
    labels = classify_behavior("United States", "USA", "USA")
    assert labels["critic_position_relation"] == "different_nonrefusal_candidate"
    assert labels["behavior_schema_version"] == "lexical_v2"
    assert labels["behavior_matching_basis"] == "normalized_text_not_semantic"
