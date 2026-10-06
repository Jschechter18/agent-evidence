import pytest

from mas_sae.experiments.artifacts import sha256_text
from mas_sae.evaluation.critic_retention import candidate_status, paired_retention, retention_report


def row(blind="Paris", a1="London", advocated="Paris", **extra):
    return {"question_id": "q", "status": "completed", "critic_blind_answer": blind,
            "solver_attempt_1": a1, "attempt1_sha256": sha256_text(a1), "critic_advocated_answer": advocated, "critic_noncommittal": False, **extra}


@pytest.mark.parametrize("fields, blind_kind, relation, outcome", [
    ({}, "answer", "different", "retained_own_position"),
    ({"advocated": "London"}, "answer", "different", "moved_to_solver_a1"),
    ({"advocated": "Berlin"}, "answer", "different", "third_position"),
    ({"advocated": "Paris France"}, "answer", "different", "unclear"),
    ({"advocated": "I cannot determine"}, "answer", "different", "unclear"),
    ({"critic_noncommittal": True}, "answer", "different", "unclear"),
    ({"blind": "London"}, "answer", "same", None),
    ({"blind": "London UK"}, "answer", "containment", None),
    ({"a1": "unknown"}, "answer", "solver_a1_not_answer_like", None),
    ({"blind": "not enough information"}, "nonanswer", None, None),
    ({"status": "failed"}, "stage_failed", None, None),
])
def test_candidate_status_branches(fields, blind_kind, relation, outcome):
    status = candidate_status(row(**fields))
    assert (status["blind_kind"], status["blind_vs_solver_a1"], status["outcome"]) == (blind_kind, relation, outcome)
    assert status["eligible"] is (outcome is not None)


def test_missing_status_fails_closed():
    unlabelled = row()
    del unlabelled["status"]
    assert candidate_status(unlabelled)["blind_kind"] == "stage_failed"


def test_paired_counts_keep_per_arm_eligibility_visible():
    a = [candidate_status({**row(), "question_id": q}) for q in ("q1", "q2", "q3")]
    b = [candidate_status({**row(advocated="London"), "question_id": "q1"}),
         candidate_status({**row(blind="London"), "question_id": "q2"})]
    paired = paired_retention(a, b, name_a="same", name_b="cross")
    assert paired["n_jointly_eligible"] == 1
    assert paired["n_eligible_by_arm"] == {"same": 3, "cross": 1}
    assert paired["counts"] == {"both_retained": 0, "same_only_retained": 1, "cross_only_retained": 0,
                                "neither_retained": 0}


def test_report_requires_exactly_two_arms():
    with pytest.raises(ValueError, match="exactly two"):
        retention_report({"only": [row()]})


@pytest.mark.parametrize("metadata", [{}, {"critic_noncommittal": None}, {"critic_noncommittal": 0}])
def test_missing_or_nonboolean_noncommittal_cannot_count_as_retained(metadata):
    candidate = row()
    del candidate["critic_noncommittal"]
    candidate.update(metadata)
    status = candidate_status(candidate)
    assert status["eligible"] is True
    assert status["outcome"] == "unclear"


def test_binary_pairs_exclude_unclear_but_preserve_arm_outcomes_and_denominators():
    outcomes = [("Paris", "Paris"), ("Paris", "London"), ("Berlin", "Paris"),
                ("London", "Berlin"), ("Paris France", "Paris"),
                ("Paris", "Paris France"), ("Paris France", "Paris France")]
    arms = {"same": [], "cross": []}
    for i, (a, b) in enumerate(outcomes):
        arms["same"].append(row(advocated=a, question_id=f"q{i}"))
        arms["cross"].append(row(advocated=b, question_id=f"q{i}"))
    arms["same"].append(row(question_id="unpaired"))
    report = retention_report(arms)
    paired = report["paired"]
    assert paired["n_jointly_eligible"] == 7
    assert paired["n_jointly_classifiable_for_binary_retention"] == 4
    assert paired["n_jointly_eligible_excluded_unclear"] == 3
    assert paired["n_eligible_by_arm"] == {"same": 8, "cross": 7}
    assert paired["counts"] == {"both_retained": 1, "same_only_retained": 1,
                                "cross_only_retained": 1, "neither_retained": 1}
    assert sum(paired["counts"].values()) == paired["n_jointly_classifiable_for_binary_retention"]
    for arm, n in (("same", 8), ("cross", 7)):
        summary = report["arms"][arm]["disagreement_outcomes"]
        assert summary["n"] == n
        assert summary["counts"]["unclear"] == 2
        assert sum(summary["counts"].values()) == n


@pytest.mark.parametrize("arm", ["same", "cross"])
def test_pairing_rejects_duplicate_question_ids_even_when_ineligible(arm):
    arms = {"same": [row()], "cross": [row()]}
    arms[arm].append(row(status="failed"))
    with pytest.raises(ValueError, match="duplicate question_id"):
        retention_report(arms)


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_pairing_rejects_different_saved_attempt1_even_when_ineligible(status):
    with pytest.raises(ValueError, match="attempt1_sha256 differs across arms"):
        retention_report({"same": [row()], "cross": [row(a1="Berlin", status=status)]})


@pytest.mark.parametrize("arm", ["same", "cross"])
@pytest.mark.parametrize("field", ["question_id", "attempt1_sha256"])
@pytest.mark.parametrize("value", [None, "", "missing"])
def test_pairing_requires_question_and_attempt1_identity(arm, field, value):
    arms = {"same": [row()], "cross": [row()]}
    if value == "missing":
        del arms[arm][0][field]
    else:
        arms[arm][0][field] = value
    with pytest.raises(ValueError, match=field):
        retention_report(arms)


def test_unpaired_rows_still_require_attempt1_identity():
    candidate = row(question_id="unpaired")
    del candidate["attempt1_sha256"]
    with pytest.raises(ValueError, match="attempt1_sha256"):
        retention_report({"same": [candidate], "cross": []})
