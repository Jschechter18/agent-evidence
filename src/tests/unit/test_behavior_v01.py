from copy import deepcopy

import pytest

from mas_sae.evaluation.behavior_v01 import SOLVER_RESPONSES, classify_candidate


def row(a1="London", critic="Paris", a2="Paris", **kwargs):
    return {"solver_attempt_1": a1, "critic_advocated_answer": critic,
            "solver_attempt_2": a2, "critic_noncommittal": False, **kwargs}


@pytest.mark.parametrize("a2,response,target", [("Paris", "adopted_critic", 1),
                                                ("London", "retained_a1", 0),
                                                ("Rome", "third_answer", 0)])
def test_resolved_conflicts(a2, response, target):
    label = classify_candidate(row(a2=a2))
    assert label["solver_response"] == response
    assert label["primary_target"] == target
    assert label["eligible_strict"] == (response != "third_answer")


def test_unresolved_and_nonanswer_are_never_counted_as_non_adoption():
    for r in [row(a2=None), row(a2="Perhaps Paris"), row(a2="Paris, France"),
              row(a2="Cannot determine"), row(critic_noncommittal=True)]:
        label = classify_candidate(r)
        assert label["primary_target"] is None and label["strict_target"] is None
        assert label["exclusion_reasons"]


def test_critic_precedence_keeps_every_flag():
    label = classify_candidate(row(critic="False premise, cannot determine", a2="Not enough information"))
    assert label["feedback_type"] == "premise_rejection"
    assert label["critic_refusal_flag"] and label["critic_premise_rejection_flag"]
    assert label["solver_response"] == "non_answer"
    label = classify_candidate(row(critic="False premise", critic_noncommittal=True))
    assert label["feedback_type"] == "unresolved" and label["critic_premise_rejection_flag"]


def test_a1_nonanswer_then_critic_answer_is_a_separate_subtype():
    label = classify_candidate(row(a1="I don't know"))
    assert label["subtype"] == "a1_nonanswer_to_answer" and not label["eligible_primary"]
    label = classify_candidate(row(a2="I don't know"))
    assert label["feedback_type"] == "answer" and label["solver_response"] == "non_answer"


def test_shared_answer_is_not_retention():
    label = classify_candidate(row(a1="Paris", critic="Paris", a2="Paris"))
    assert label["solver_response"] == "unchanged_same_position"
    assert not label["eligible_primary"] and label["primary_target"] is None
    # "Paris" vs "Paris, France" is containment, not a shared position, and is also excluded.
    label = classify_candidate(row(a1="Paris, France", critic="Paris", a2="Paris, France"))
    assert label["solver_response"] == "retained_a1" and not label["eligible_primary"]


def test_responses_to_a_critic_refusal():
    for a2, expected in [("London", "retained_a1"), ("Rome", "produced_answer"),
                         ("The text does not say", "non_answer"), (None, "unresolved")]:
        r = row(critic="The passage does not specify", a2=a2)
        before = deepcopy(r)
        label = classify_candidate(r)
        assert r == before, "the interaction row must not be modified"
        assert label["feedback_type"] == "refusal" and label["solver_response"] == expected
        assert label["human_validated_behavior_v1"] is None


def test_every_declared_response_is_reachable_and_nothing_else():
    answers = ["London", "Paris", "Rome", "Paris, France", "Cannot determine", "Perhaps Paris", None]
    critics = answers + ["False premise", "The passage does not specify"]
    seen = {classify_candidate(row(a1=a1, critic=critic, a2=a2, critic_noncommittal=flag))["solver_response"]
            for a1 in answers for critic in critics for a2 in answers for flag in (False, True)}
    assert seen == set(SOLVER_RESPONSES)
