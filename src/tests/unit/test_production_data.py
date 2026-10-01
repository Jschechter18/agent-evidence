import pytest
import torch

from mas_sae.data.production import (
    authorize_rows,
    canonical_manifest,
    check_scan_repeats_full_run,
    load_labeled_rows,
    load_production_activations,
    read_manifest,
    split_audit,
    write_manifest,
)
from mas_sae.evaluation.behavior_v01 import classify_candidate, write_labels
from mas_sae.experiments.records import write_jsonl


def row(qid="q", split="discovery", i=0, source="train"):
    return {"question_id": qid, "source_split": source, "experiment_split": split,
            "attempt1_activation_index": i, "attempt2_activation_index": i}


def test_full_run_split_wins_over_scan_split():
    full, scan = row(split="validation"), row(split="discovery")
    manifest = canonical_manifest([full], [scan])
    assert manifest["q"]["canonical_split"] == "validation" and manifest["q"]["scan_split"] == "discovery"
    with pytest.raises(ValueError, match="requires discovery"):
        authorize_rows([scan], manifest, purpose="fit")
    assert authorize_rows([scan], manifest, purpose="evaluate") == [scan]


def test_purpose_must_match_partition_and_questions_cannot_repeat():
    rows = [row("a"), row("b", split="intervention", i=1)]
    manifest = canonical_manifest(rows)
    assert authorize_rows(rows[:1], manifest, purpose="tune") == rows[:1]
    with pytest.raises(ValueError, match="requires discovery"):
        authorize_rows(rows, manifest, purpose="fit")
    with pytest.raises(ValueError, match="repeated"):
        authorize_rows([rows[0], rows[0]], manifest, purpose="fit")
    with pytest.raises(ValueError, match="purpose must be"):
        authorize_rows(rows[:1], manifest, purpose="anything")


def test_development_exposure_is_flagged_and_survives_csv(tmp_path):
    full = [row("a"), row("b", split="validation", i=1), row("c", split="validation", i=2)]
    manifest = canonical_manifest(full, scan_rows=[row("a")], qc100_ids=["b"])
    assert [manifest[q]["in_development_data"] for q in "abc"] == [True, True, False]
    assert manifest["b"]["in_qc100"] and not manifest["a"]["in_qc100"]
    write_manifest(tmp_path / "m.csv", manifest)
    assert read_manifest(tmp_path / "m.csv") == manifest


def test_manifest_rejects_duplicates_and_cross_source_questions():
    with pytest.raises(ValueError, match="one episode per question"):
        canonical_manifest([row(), row(i=1)])
    with pytest.raises(ValueError, match="crosses source splits"):
        canonical_manifest([row()], [row(source="validation")])
    with pytest.raises(ValueError, match="missing from the full run"):
        canonical_manifest([row()], [row("other")])


def save_activations(root, values):
    layer_dir = root / "layer_08"
    layer_dir.mkdir(parents=True)
    torch.save(torch.tensor(values), layer_dir / "train_attempt2.pt")


def test_activations_follow_row_order_and_respect_the_guard(tmp_path):
    rows = [row("a", i=0), row("b", i=1), row("c", split="intervention", i=2)]
    manifest = canonical_manifest(rows)
    save_activations(tmp_path, [[1., 2.], [3., 4.], [5., 6.]])
    result = load_production_activations([rows[1], rows[0]], manifest, tmp_path, purpose="fit", layer=8)
    assert result.tolist() == [[3., 4.], [1., 2.]]
    with pytest.raises(ValueError, match="requires discovery"):
        load_production_activations(rows, manifest, tmp_path, purpose="fit", layer=8)


def test_loader_never_returns_another_questions_activation(tmp_path):
    # Full run: a, b, c at rows 0, 1, 2. Scan re-ran b and c at rows 0, 1.
    full = [row("a", i=0), row("b", i=1), row("c", i=2)]
    scan = [row("b", i=0), row("c", i=1)]
    manifest = canonical_manifest(full, scan)
    save_activations(tmp_path / "full", [[10.], [11.], [12.]])
    save_activations(tmp_path / "scan", [[21.], [22.]])

    assert load_production_activations(scan, manifest, tmp_path / "scan", purpose="fit", layer=8,
                                       run="scan").tolist() == [[21.], [22.]]
    # Scan rows read against the full run would return question a's and b's activations.
    with pytest.raises(ValueError, match="does not match the full run"):
        load_production_activations(scan, manifest, tmp_path / "full", purpose="fit", layer=8)
    # Declaring the scan run but pointing at the full run's files is also refused.
    with pytest.raises(ValueError, match="from a different run"):
        load_production_activations(scan, manifest, tmp_path / "full", purpose="fit", layer=8, run="scan")
    # A row whose index was altered is refused.
    with pytest.raises(ValueError, match="does not match the full run"):
        load_production_activations([{**full[0], "attempt2_activation_index": 1}], manifest,
                                    tmp_path / "full", purpose="fit", layer=8)


def test_manifest_rejects_bad_activation_indices():
    with pytest.raises(ValueError, match="reuses activation row"):
        canonical_manifest([row("a", i=0), row("b", i=0)])
    with pytest.raises(ValueError, match="non-negative integer"):
        canonical_manifest([row("a", i=-1)])
    with pytest.raises(ValueError, match="non-negative integer"):
        canonical_manifest([row("a", i="0")])
    with pytest.raises(ValueError, match="scan run reuses"):
        canonical_manifest([row("a", i=0), row("b", i=1)], [row("a", i=0), row("b", i=0)])
    # The same index in different source splits is a different file, so it is fine.
    canonical_manifest([row("a", i=0), row("b", i=0, source="validation")])


def test_loader_rejects_a_row_with_the_wrong_source_split(tmp_path):
    rows = [row("a", i=0), row("b", i=0, source="validation")]
    manifest = canonical_manifest(rows)
    save_activations(tmp_path, [[1.]])
    torch.save(torch.tensor([[2.]]), tmp_path / "layer_08" / "validation_attempt2.pt")
    wrong = {**rows[1], "source_split": "train"}
    with pytest.raises(ValueError, match="source split does not match"):
        load_production_activations([wrong], manifest, tmp_path, purpose="fit", layer=8)


def test_loader_reuses_activation_store_validation(tmp_path):
    rows = [row("a", i=0)]
    manifest = canonical_manifest(rows)
    layer_dir = tmp_path / "layer_08"
    layer_dir.mkdir()
    torch.save(torch.tensor([1., 2.]), layer_dir / "train_attempt2.pt")  # 1-D, not a matrix
    with pytest.raises(ValueError, match="2-dimensional"):
        load_production_activations(rows, manifest, tmp_path, purpose="fit", layer=8)
    with pytest.raises(FileNotFoundError):
        load_production_activations(rows, manifest, tmp_path, purpose="fit", layer=9)


def test_split_audit_counts_scan_conflicts():
    full = [row("a"), row("b", split="validation", i=1), row("c", split="intervention", i=2)]
    scan = [row("a"), row("b", split="discovery", i=1)]
    audit = split_audit(canonical_manifest(full, scan, qc100_ids=["c"]))
    assert audit["scan_questions_also_in_full_run"] == 2
    assert audit["scan_questions_with_a_different_split"] == 1
    assert audit["scan_discovery_but_held_out_in_full_run"] == 1
    assert audit["in_development_data_by_canonical_split"] == {
        "discovery": 1, "validation": 1, "intervention": 1}


def test_scan_text_must_match_the_full_run():
    answers = {"solver_attempt_1": "London", "critic_advocated_answer": "Paris",
               "solver_attempt_2": "Paris", "critic_noncommittal": False}
    full = [{**row("a"), **answers}]
    check_scan_repeats_full_run(full, [{**row("a"), **answers}])
    with pytest.raises(ValueError, match="disagree on solver_attempt_2"):
        check_scan_repeats_full_run(full, [{**row("a"), **answers, "solver_attempt_2": "Rome"}])
    # Same answer text but a different Critic flag would change the label.
    with pytest.raises(ValueError, match="disagree on critic_noncommittal"):
        check_scan_repeats_full_run(full, [{**row("a"), **answers, "critic_noncommittal": True}])


def test_load_labeled_rows_joins_labels_and_manifest(tmp_path):
    def episode(qid, a2, split, source):
        return {**row(qid, split=split, source=source), "solver_attempt_1": "London",
                "critic_advocated_answer": "Paris", "solver_attempt_2": a2,
                "critic_noncommittal": False, "solver_behavior": "lexical"}

    interactions = {"train": [episode("a", "Paris", "discovery", "train")],
                    "validation": [episode("b", "Cannot determine", "validation", "validation")]}
    results, package = tmp_path / "results", tmp_path / "package"
    package.mkdir()
    labelled = []
    for source, episodes in interactions.items():
        write_jsonl(results / source / "interactions.jsonl", episodes)
        for item in episodes:
            labelled.append({**item, "canonical_split": item["experiment_split"],
                             "label": classify_candidate(item)})
    write_manifest(package / "split_manifest.csv", canonical_manifest(labelled))
    write_labels(package / "labels.csv", labelled)

    rows, manifest = load_labeled_rows(results, package)
    assert [r["canonical_split"] for r in rows] == ["discovery", "validation"]
    assert rows[0]["label"]["eligible_primary"] is True
    assert rows[0]["label"]["primary_target"] == 1
    assert rows[0]["label"]["exclusion_reasons"] == []
    assert rows[1]["label"]["eligible_primary"] is False
    assert rows[1]["label"]["primary_target"] is None
    assert rows[1]["label"]["exclusion_reasons"] == ["a2_nonanswer"]
    # Reading restores exactly what was written, for every label field.
    for loaded, original in zip(rows, labelled):
        assert {field: loaded["label"][field] for field in original["label"]} == original["label"]
    assert authorize_rows(rows[:1], manifest, purpose="fit") == rows[:1]
