import pytest

from mas_sae.evaluation.behavior_qc import (
    ALLOWED_ANSWERS,
    agreement,
    check_qc_packet,
    choose_qc_rows,
    write_annotator_files,
)
from mas_sae.evaluation.behavior_v01 import SOLVER_RESPONSES, classify_candidate


def episode(number, a2, split="discovery"):
    row = {"question_id": f"q{number}", "question": f"Question {number}?",
           "solver_attempt_1": "London", "critic_advocated_answer": "Paris",
           "critic_feedback": "The reviewer advocates: Paris", "solver_attempt_2": a2,
           "critic_noncommittal": False, "canonical_split": split}
    row["label"] = classify_candidate(row)
    return row


def annotation(**values):
    return {"question": "Q", "feedback_type": "", "solver_response": "",
            "eligible_primary": "", **values}


@pytest.fixture
def episodes():
    kept = [episode(n, "London") for n in range(3)]
    adopted = [episode(n, "Paris") for n in range(3, 60)]
    held_out = [episode(n, "London", split="validation") for n in range(60, 70)]
    return kept + adopted + held_out


def test_packet_takes_every_kept_case_and_only_discovery(episodes):
    chosen, key, sampling = choose_qc_rows(episodes, seed=42)
    assert all(row["canonical_split"] == "discovery" for row in chosen)
    assert sampling["selected"] == {"adopted_critic": 40, "retained_a1": 3}
    assert sampling["population"] == {"adopted_critic": 57, "retained_a1": 3}
    assert sum(entry["second_annotator"] for entry in key) == 13
    assert [entry["review_id"] for entry in key[:2]] == ["Q0001", "Q0002"]
    # Same seed, same packet.
    assert choose_qc_rows(episodes, seed=42)[1] == key


def test_written_packet_is_blind_and_matches_the_run(tmp_path, episodes):
    chosen, key, _ = choose_qc_rows(episodes, seed=42)
    paragraphs = {row["question_id"]: [{"idx": 0, "title": "T", "paragraph_text": "text"}]
                  for row in chosen}
    write_annotator_files(tmp_path, chosen, key, paragraphs)
    rows_by_question = {row["question_id"]: row for row in episodes}
    check_qc_packet(tmp_path, key, rows_by_question)

    header = (tmp_path / "annotator_a.csv").read_text().splitlines()[0]
    assert "stratum" not in header and "question_id" not in header and "split" not in header

    # A packet row whose text no longer matches the run is refused.
    changed = dict(rows_by_question)
    changed[key[0]["question_id"]] = {**changed[key[0]["question_id"]], "solver_attempt_2": "X"}
    with pytest.raises(ValueError, match="differs from the production run"):
        check_qc_packet(tmp_path, key, changed)


def test_second_annotator_file_gets_the_same_checks(tmp_path, episodes):
    chosen, key, _ = choose_qc_rows(episodes, seed=42)
    paragraphs = {row["question_id"]: [{"idx": 0, "title": "T", "paragraph_text": "text"}]
                  for row in chosen}
    rows_by_question = {row["question_id"]: row for row in episodes}
    second_file = tmp_path / "annotator_b.csv"

    write_annotator_files(tmp_path, chosen, key, paragraphs)
    original = second_file.read_text()

    second_file.write_text(original.replace(",Paris,", ",Rome,", 1))
    with pytest.raises(ValueError, match="annotator_b.csv: text differs"):
        check_qc_packet(tmp_path, key, rows_by_question)

    lines = original.splitlines()
    second_file.write_text("\n".join([lines[0], lines[1].rstrip(",") + ",prefilled"] + lines[2:]))
    with pytest.raises(ValueError, match="annotator_b.csv: annotation columns must start empty"):
        check_qc_packet(tmp_path, key, rows_by_question)


def test_blank_annotations_are_not_agreement():
    result = agreement({"1": annotation()}, {"1": annotation()})
    assert result["fields"]["feedback_type"]["paired_completed"] == 0
    assert result["fields"]["feedback_type"]["raw_agreement"] is None


def test_agreement_and_disagreement_are_counted():
    a = {"1": annotation(solver_response="retained_a1"),
         "2": annotation(solver_response="adopted_critic")}
    b = {"1": annotation(solver_response="retained_a1"),
         "2": annotation(solver_response="third_answer")}
    field = agreement(a, b)["fields"]["solver_response"]
    assert field["paired_completed"] == 2 and field["raw_agreement"] == 0.5
    assert field["disagreement_ids"] == ["2"]


def test_typos_and_mismatched_evidence_are_rejected():
    with pytest.raises(ValueError, match="Invalid"):
        agreement({"1": annotation(feedback_type="asnwer")}, {})
    with pytest.raises(ValueError, match="Different"):
        agreement({"1": annotation()}, {"1": annotation(question="Different Q")})


def test_annotators_may_use_every_classifier_response():
    assert ALLOWED_ANSWERS["solver_response"] == set(SOLVER_RESPONSES)
