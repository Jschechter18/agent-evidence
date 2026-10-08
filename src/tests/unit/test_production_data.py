from collections import Counter
import csv
import json
from pathlib import Path
import random

import pytest
import torch

from mas_sae.data.production import (
    INTERVENTION_IDS_FILE,
    MANIFEST_FIELDS,
    PARTITION_FIELDS,
    PARTITIONS,
    PARTITIONS_FILE,
    assign_partitions,
    authorize_rows,
    build_partition_manifest,
    check_manifest_matches_frozen,
    check_scan_repeats_full_run,
    export_partition_activations,
    load_labeled_rows,
    load_production_activations,
    partition_coverage,
    read_manifest,
    read_partitions,
    split_audit,
    write_manifest,
    write_partitions,
)
from mas_sae.evaluation.behavior_v01 import classify_candidate, write_labels
from mas_sae.experiments.records import read_jsonl, write_jsonl

PROPORTIONS = {"train": 0.6, "test": 0.2, "intervention": 0.2}


def row(qid="2hop__1_1", split="discovery", i=0, source="train"):
    return {"question_id": qid, "source_split": source, "experiment_split": split,
            "attempt1_activation_index": i, "attempt2_activation_index": i}


def train_rows(n, prefix="2hop"):
    return [row(f"{prefix}__{k}_{k}", i=k) for k in range(n)]


def manifest_for(full, scan=(), qc100_ids=(), seed=0):
    return build_partition_manifest(full, scan, qc100_ids, proportions=PROPORTIONS, seed=seed)


# ------------------------------------------------------------------ the partition rule

def test_every_question_gets_exactly_one_partition_and_sets_are_disjoint():
    full = train_rows(20) + [row(f"3hop1__{k}_{k}", i=k, source="validation") for k in range(5)]
    manifest = manifest_for(full)
    assert set(manifest) == {r["question_id"] for r in full}
    by_partition = {p: {q for q, m in manifest.items() if m["partition"] == p} for p in PARTITIONS}
    assert all(m["partition"] in PARTITIONS for m in manifest.values())
    ids = list(by_partition.values())
    assert sum(len(s) for s in ids) == len(manifest)          # cover, no overlap
    assert set.union(*ids) == set(manifest)


def test_source_validation_is_the_validation_partition_and_source_train_never_is():
    full = train_rows(10) + [row(f"2hop__v{k}_{k}", i=k, source="validation") for k in range(4)]
    manifest = manifest_for(full)
    for m in manifest.values():
        assert (m["partition"] == "validation") == (m["source_split"] == "validation")


def test_assignment_depends_only_on_the_question_set_proportions_and_seed():
    ids = [f"2hop__{k}_{k}" for k in range(30)] + [f"3hop1__{k}_{k}" for k in range(10)]
    shuffled = list(ids)
    random.Random(7).shuffle(shuffled)
    first = assign_partitions({"train": ids}, proportions=PROPORTIONS, seed=3)
    second = assign_partitions({"train": shuffled}, proportions=PROPORTIONS, seed=3)
    other_seed = assign_partitions({"train": ids}, proportions=PROPORTIONS, seed=4)
    assert first == second
    assert first != other_seed


def test_equal_proportions_in_any_key_order_give_the_same_assignment():
    ids = {"train": [f"2hop__{k}_{k}" for k in range(100)]}
    forward = assign_partitions(ids, proportions={"train": .8, "test": .1, "intervention": .1}, seed=42)
    reversed_keys = assign_partitions(ids, proportions={"intervention": .1, "test": .1, "train": .8}, seed=42)
    assert forward == reversed_keys


def test_assignment_keeps_hop_mix_and_proportions():
    ids = [f"2hop__{k}_{k}" for k in range(20)] + [f"4hop1__{k}_{k}" for k in range(10)]
    assignment = assign_partitions({"train": ids}, proportions=PROPORTIONS, seed=1)
    counts = {}
    for qid, partition in assignment.items():
        counts[(qid[:4], partition)] = counts.get((qid[:4], partition), 0) + 1
    assert counts[("2hop", "train")] == 12 and counts[("2hop", "test")] == 4
    assert counts[("4hop", "train")] == 6 and counts[("4hop", "intervention")] == 2


def test_assign_partitions_rejects_wrong_names_sources_and_crossing_ids():
    with pytest.raises(ValueError, match="must name exactly"):
        assign_partitions({"train": ["2hop__1_1"]}, proportions={"train": 0.5, "test": 0.5}, seed=0)
    with pytest.raises(ValueError, match="unknown source splits"):
        assign_partitions({"dev": ["2hop__1_1"]}, proportions=PROPORTIONS, seed=0)
    with pytest.raises(ValueError, match="both source splits"):
        assign_partitions({"train": ["2hop__1_1"], "validation": ["2hop__1_1"]},
                          proportions=PROPORTIONS, seed=0)


# ------------------------------------------------------------------ the manifest

def test_manifest_keeps_history_and_flags_development_exposure(tmp_path):
    full = train_rows(5)
    scan = [row("2hop__0_0", split="intervention", i=0)]
    manifest = manifest_for(full, scan, qc100_ids=["2hop__1_1"])
    assert manifest["2hop__0_0"]["collection_split"] == "discovery"
    assert manifest["2hop__0_0"]["scan_split"] == "intervention"
    assert manifest["2hop__0_0"]["scan_index"] == 0
    assert [manifest[f"2hop__{k}_{k}"]["in_development_data"] for k in range(3)] == [True, True, False]
    assert manifest["2hop__1_1"]["in_qc100"] and not manifest["2hop__0_0"]["in_qc100"]
    assert all(m["hop_group"] == "2hop" for m in manifest.values())
    assert list(manifest["2hop__0_0"]) == MANIFEST_FIELDS
    write_manifest(tmp_path / "m.csv", manifest)
    assert read_manifest(tmp_path / "m.csv") == manifest


def test_manifest_rejects_duplicates_bad_splits_and_cross_source_questions():
    with pytest.raises(ValueError, match="one episode per question"):
        manifest_for([row(), row(i=1)])
    with pytest.raises(ValueError, match="crosses source splits"):
        manifest_for([row()], [row(source="validation")])
    with pytest.raises(ValueError, match="missing from the full run"):
        manifest_for([row()], [row("2hop__9_9")])
    with pytest.raises(ValueError, match="unknown experiment split"):
        manifest_for([row(split="holdout")])
    with pytest.raises(ValueError, match="unknown source split"):
        manifest_for([row(source="test")])


def test_manifest_rejects_bad_activation_indices():
    with pytest.raises(ValueError, match="reuses activation row"):
        manifest_for([row("2hop__1_1", i=0), row("2hop__2_2", i=0)])
    with pytest.raises(ValueError, match="non-negative integer"):
        manifest_for([row(i=-1)])
    with pytest.raises(ValueError, match="non-negative integer"):
        manifest_for([row(i="0")])
    with pytest.raises(ValueError, match="scan run reuses"):
        manifest_for([row("2hop__1_1", i=0), row("2hop__2_2", i=1)],
                     [row("2hop__1_1", i=0), row("2hop__2_2", i=0)])
    # The same index in different source splits is a different file, so it is fine.
    manifest_for([row("2hop__1_1", i=0), row("2hop__2_2", i=0, source="validation")])


def test_split_audit_reports_partitions_and_exposure():
    full = train_rows(10)
    scan = [row("2hop__0_0", split="validation", i=0)]
    manifest = manifest_for(full, scan, qc100_ids=["2hop__1_1"])
    audit = split_audit(manifest)
    assert audit["questions"] == 10
    assert sum(audit["by_partition"].values()) == 10
    assert sum(audit["by_partition_source_and_hop"].values()) == 10
    assert audit["scan_questions_also_in_full_run"] == 1
    assert audit["scan_questions_with_a_different_collection_split"] == 1
    assert sum(audit["in_development_data_by_partition"].values()) == 2
    assert sum(audit["qc100_by_partition"].values()) == 1


# ------------------------------------------------------------------ the guard

def partition_examples():
    """A manifest with at least one question in every partition, and a row per partition."""
    full = train_rows(15) + [row("2hop__v0_0", i=0, source="validation")]
    manifest = manifest_for(full)
    by_partition = {}
    for r in full:
        by_partition.setdefault(manifest[r["question_id"]]["partition"], r)
    assert set(by_partition) == set(PARTITIONS)
    return full, manifest, by_partition


def test_each_purpose_reads_exactly_one_partition():
    _, manifest, by_partition = partition_examples()
    allowed = {"fit": "train", "tune": "validation", "evaluate": "test", "intervene": "intervention"}
    for purpose, partition in allowed.items():
        assert authorize_rows([by_partition[partition]], manifest, purpose=purpose)
        for other in PARTITIONS:
            if other != partition:
                with pytest.raises(ValueError, match=f"requires {partition}"):
                    authorize_rows([by_partition[other]], manifest, purpose=purpose)
    with pytest.raises(ValueError, match="repeated"):
        authorize_rows([by_partition["train"]] * 2, manifest, purpose="fit")
    with pytest.raises(ValueError, match="purpose must be"):
        authorize_rows([by_partition["train"]], manifest, purpose="anything")


def test_collection_time_split_does_not_override_the_partition():
    full, manifest, _ = partition_examples()
    scan_row = {**full[0], "experiment_split": "intervention"}   # scan called it something else
    partition = manifest[scan_row["question_id"]]["partition"]
    purpose = {"train": "fit", "validation": "tune", "test": "evaluate", "intervention": "intervene"}[partition]
    assert authorize_rows([scan_row], manifest, purpose=purpose) == [scan_row]


# ------------------------------------------------------------------ the loader

def save_activations(root, values, name="train_attempt2.pt"):
    layer_dir = root / "layer_08"
    layer_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(values), layer_dir / name)


def test_activations_follow_row_order_and_respect_the_guard(tmp_path):
    full, manifest, by_partition = partition_examples()
    source_train = [r for r in full if r["source_split"] == "train"]
    save_activations(tmp_path, [[float(r["attempt2_activation_index"]), 1.] for r in source_train])
    fit_rows = [r for r in source_train if manifest[r["question_id"]]["partition"] == "train"]
    result = load_production_activations(fit_rows[::-1], manifest, tmp_path, purpose="fit", layer=8)
    assert result[:, 0].tolist() == [float(r["attempt2_activation_index"]) for r in fit_rows[::-1]]
    with pytest.raises(ValueError, match="requires train"):
        load_production_activations([by_partition["test"]], manifest, tmp_path, purpose="fit", layer=8)


def test_loader_never_returns_another_questions_activation(tmp_path):
    # Full run: a, b, c at rows 0, 1, 2. Scan re-ran b and c at rows 0, 1.
    full = [row("2hop__1_1", i=0), row("2hop__2_2", i=1), row("2hop__3_3", i=2)]
    scan = [row("2hop__2_2", i=0), row("2hop__3_3", i=1)]
    manifest = build_partition_manifest(full, scan, proportions={"train": 0.98, "test": 0.01,
                                                                 "intervention": 0.01}, seed=0)
    assert all(m["partition"] == "train" for m in manifest.values())
    save_activations(tmp_path / "full", [[10.], [11.], [12.]])
    save_activations(tmp_path / "scan", [[21.], [22.]])

    assert load_production_activations(scan, manifest, tmp_path / "scan", purpose="fit", layer=8,
                                       run="scan").tolist() == [[21.], [22.]]
    with pytest.raises(ValueError, match="does not match the full run"):
        load_production_activations(scan, manifest, tmp_path / "full", purpose="fit", layer=8)
    with pytest.raises(ValueError, match="from a different run"):
        load_production_activations(scan, manifest, tmp_path / "full", purpose="fit", layer=8, run="scan")
    with pytest.raises(ValueError, match="does not match the full run"):
        load_production_activations([{**full[0], "attempt2_activation_index": 1}], manifest,
                                    tmp_path / "full", purpose="fit", layer=8)


def test_loader_rejects_a_row_with_the_wrong_source_split(tmp_path):
    full = [row("2hop__1_1", i=0), row("2hop__2_2", i=0, source="validation")]
    manifest = build_partition_manifest(full, proportions={"train": 0.98, "test": 0.01,
                                                           "intervention": 0.01}, seed=0)
    save_activations(tmp_path, [[1.]])
    save_activations(tmp_path, [[2.]], name="validation_attempt2.pt")
    wrong = {**full[1], "source_split": "train"}
    with pytest.raises(ValueError, match="source split does not match"):
        load_production_activations([wrong], manifest, tmp_path, purpose="tune", layer=8)


def test_loader_reuses_activation_store_validation(tmp_path):
    full = [row("2hop__1_1", i=0)]
    manifest = build_partition_manifest(full, proportions={"train": 0.98, "test": 0.01,
                                                           "intervention": 0.01}, seed=0)
    layer_dir = tmp_path / "layer_08"
    layer_dir.mkdir()
    torch.save(torch.tensor([1., 2.]), layer_dir / "train_attempt2.pt")  # 1-D, not a matrix
    with pytest.raises(ValueError, match="2-dimensional"):
        load_production_activations(full, manifest, tmp_path, purpose="fit", layer=8)
    with pytest.raises(FileNotFoundError):
        load_production_activations(full, manifest, tmp_path, purpose="fit", layer=9)


def test_scan_text_must_match_the_full_run():
    answers = {"solver_attempt_1": "London", "critic_advocated_answer": "Paris",
               "solver_attempt_2": "Paris", "critic_noncommittal": False}
    full = [{**row(), **answers}]
    check_scan_repeats_full_run(full, [{**row(), **answers}])
    with pytest.raises(ValueError, match="disagree on solver_attempt_2"):
        check_scan_repeats_full_run(full, [{**row(), **answers, "solver_attempt_2": "Rome"}])
    with pytest.raises(ValueError, match="disagree on critic_noncommittal"):
        check_scan_repeats_full_run(full, [{**row(), **answers, "critic_noncommittal": True}])


def test_load_labeled_rows_joins_labels_and_manifest(tmp_path):
    def episode(qid, a2, source, i):
        return {**row(qid, source=source, i=i), "solver_attempt_1": "London",
                "critic_advocated_answer": "Paris", "solver_attempt_2": a2,
                "critic_noncommittal": False, "solver_behavior": "lexical"}

    interactions = {"train": [episode("2hop__1_1", "Paris", "train", 0)],
                    "validation": [episode("2hop__2_2", "Cannot determine", "validation", 0)]}
    results, package = tmp_path / "results", tmp_path / "package"
    package.mkdir()
    all_rows = [r for group in interactions.values() for r in group]
    manifest = build_partition_manifest(all_rows, proportions={"train": 0.98, "test": 0.01,
                                                               "intervention": 0.01}, seed=0)
    labelled = []
    for source, episodes in interactions.items():
        write_jsonl(results / source / "interactions.jsonl", episodes)
        for item in episodes:
            labelled.append({**item, "partition": manifest[item["question_id"]]["partition"],
                             "label": classify_candidate(item)})
    write_manifest(package / "split_manifest.csv", manifest)
    write_labels(package / "labels.csv", labelled)

    rows, loaded = load_labeled_rows(results, package)
    assert [r["partition"] for r in rows] == ["train", "validation"]
    assert rows[0]["label"]["eligible_primary"] is True
    assert rows[0]["label"]["primary_target"] == 1
    assert rows[1]["label"]["eligible_primary"] is False
    assert rows[1]["label"]["primary_target"] is None
    for item, original in zip(rows, labelled):
        assert {field: item["label"][field] for field in original["label"]} == original["label"]
    assert authorize_rows(rows[:1], loaded, purpose="fit") == rows[:1]
    assert authorize_rows(rows[1:], loaded, purpose="tune") == rows[1:]


# ------------------------------------------------------------------ the export

def write_run(root, full, dim=2):
    """Collection-layout tensors for one run: rows are [index, source flag] so they are checkable."""
    for source, flag in (("train", 0.), ("validation", 1.)):
        rows = sorted((r for r in full if r["source_split"] == source), key=lambda r: r["attempt2_activation_index"])
        if not rows:
            continue
        for attempt, offset in ((1, 100.), (2, 200.)):
            save_activations(root, [[r["attempt2_activation_index"] + offset, flag] for r in rows],
                             name=f"{source}_attempt{attempt}.pt")


def exported_run(tmp_path, partitions=("train", "validation"), name="partitioned"):
    full, manifest, _ = partition_examples()
    if not (tmp_path / "src").exists():
        write_run(tmp_path / "src", full)
    summary = export_partition_activations(
        full, manifest, tmp_path / "src", run="full", layers=[8],
        export_activation_root=tmp_path / "data", export_result_root=tmp_path / "results",
        export_run_name=name, partitions=partitions)
    return full, manifest, summary


def test_export_mirrors_collection_layout_and_keeps_a1_a2_together(tmp_path):
    full, manifest, summary = exported_run(tmp_path)
    layer_dir = tmp_path / "data" / "partitioned" / "layer_08"
    for partition in ("train", "validation"):
        a1 = torch.load(layer_dir / f"{partition}_attempt1.pt")
        a2 = torch.load(layer_dir / f"{partition}_attempt2.pt")
        both = torch.load(layer_dir / f"{partition}.pt")
        records = read_jsonl(tmp_path / "results" / "partitioned" / partition / "interactions.jsonl")
        assert a1.shape[0] == a2.shape[0] == len(records) == summary[partition]["questions"]
        assert torch.equal(both, torch.cat([a1, a2]))
        for position, record in enumerate(records):
            assert record["partition"] == partition
            assert manifest[record["question_id"]]["partition"] == partition
            assert record["attempt1_activation_index"] == record["attempt2_activation_index"] == position
            assert record["sae_attempt1_index"] == position
            assert record["sae_attempt2_index"] == len(records) + position
            # Each exported vector is the original row for that question.
            original = manifest[record["question_id"]]["full_index"]
            assert record["source_activation_index"] == original and record["source_run"] == "full"
            assert a1[position].tolist() == [original + 100., float(record["source_split"] == "validation")]
            assert a2[position].tolist() == [original + 200., float(record["source_split"] == "validation")]
    # Development export: no test or intervention files anywhere, so nothing can score them.
    for sealed in ("test", "intervention"):
        assert not (layer_dir / f"{sealed}.pt").exists()
        assert not (tmp_path / "results" / "partitioned" / sealed).exists()
        assert summary[sealed]["rows"] == 0
    ids = json.loads((tmp_path / "results" / "partitioned" / INTERVENTION_IDS_FILE).read_text())
    assert ids == sorted(q for q, m in manifest.items() if m["partition"] == "intervention")
    assert summary["intervention"]["questions"] == len(ids)


def test_no_question_appears_in_two_exported_partitions(tmp_path):
    exported_run(tmp_path, partitions=PARTITIONS, name="everything")
    seen = {}
    for path in sorted((tmp_path / "results" / "everything").glob("*/interactions.jsonl")):
        for record in read_jsonl(path):
            assert record["question_id"] not in seen, (record["question_id"], seen.get(record["question_id"]), path)
            seen[record["question_id"]] = path.parent.name
    assert set(seen.values()) == set(PARTITIONS)
    assert (tmp_path / "data" / "everything" / "layer_08" / "intervention.pt").exists()


def test_test_and_intervention_are_exported_only_when_named_and_under_their_own_name(tmp_path):
    exported_run(tmp_path)                                   # development pair
    _, _, summary = exported_run(tmp_path, partitions=("test",), name="partitioned_test")
    assert summary["test"]["rows"] > 0 and summary["train"]["rows"] == 0
    assert (tmp_path / "data" / "partitioned_test" / "layer_08" / "test.pt").exists()
    assert not (tmp_path / "data" / "partitioned" / "layer_08" / "test.pt").exists()
    with pytest.raises(ValueError, match="unknown partitions"):
        exported_run(tmp_path, partitions=("holdout",), name="bad")


def test_export_refuses_an_existing_destination_so_nothing_goes_stale(tmp_path):
    exported_run(tmp_path, partitions=PARTITIONS, name="first")
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    # Same destination, fewer partitions: refused before anything is written.
    with pytest.raises(FileExistsError):
        exported_run(tmp_path, partitions=("train",), name="first")
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert after == before
    # A destination that is the source run itself is refused too.
    full, manifest, _ = partition_examples()
    with pytest.raises(FileExistsError):
        export_partition_activations(
            full, manifest, tmp_path / "src", run="full", layers=[8],
            export_activation_root=tmp_path, export_result_root=tmp_path / "results",
            export_run_name="src")


def test_export_refuses_rows_from_a_different_run(tmp_path):
    full, manifest, _ = partition_examples()
    write_run(tmp_path / "src", full)
    with pytest.raises(ValueError, match="different run"):
        export_partition_activations(
            full[:3], manifest, tmp_path / "src", run="full", layers=[8],
            export_activation_root=tmp_path / "data", export_result_root=tmp_path / "results",
            export_run_name="partitioned")


# ------------------------------------------------------------------ the frozen mapping

def frozen_from(manifest):
    return {q: {field: m[field] for field in PARTITION_FIELDS} for q, m in manifest.items()}


def recollected(full, seed):
    """The same questions collected again in another order: new activation rows per source split."""
    shuffled = list(full)
    random.Random(seed).shuffle(shuffled)
    next_index, rerun = {}, []
    for r in shuffled:
        i = next_index.get(r["source_split"], 0)
        next_index[r["source_split"]] = i + 1
        rerun.append({**r, "attempt1_activation_index": i, "attempt2_activation_index": i})
    return rerun


def test_partitions_file_holds_the_manifests_frozen_columns_and_round_trips(tmp_path):
    full = train_rows(15) + [row("2hop__v0_0", i=0, source="validation")]
    manifest = manifest_for(full)
    write_partitions(tmp_path / "p.csv", manifest)
    frozen = read_partitions(tmp_path / "p.csv")
    assert frozen == frozen_from(manifest)
    assert list(next(iter(frozen.values()))) == PARTITION_FIELDS
    # A full manifest is accepted as a frozen source too; its extra columns are ignored.
    write_manifest(tmp_path / "m.csv", manifest)
    assert read_partitions(tmp_path / "m.csv") == frozen


def test_read_partitions_rejects_malformed_mappings(tmp_path):
    def write(rows, header=PARTITION_FIELDS):
        path = tmp_path / "p.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        return path

    good = {"question_id": "2hop__1_1", "source_split": "train", "partition": "train", "hop_group": "2hop"}
    assert read_partitions(write([good])) == {"2hop__1_1": good}
    for bad, message in [
        ([good, good], "appears twice"),
        ([{**good, "source_split": "dev"}], "unknown source split"),
        ([{**good, "partition": "holdout"}], "unknown partition"),
        ([{**good, "hop_group": "3hop"}], "does not match the id"),
        ([{**good, "source_split": "validation"}], "only MuSiQue-validation"),
        ([{**good, "partition": "validation"}], "only MuSiQue-validation"),
        ([], "no questions"),
    ]:
        with pytest.raises(ValueError, match=message):
            read_partitions(write(bad))
    with pytest.raises(ValueError, match="missing columns"):
        read_partitions(write([{"question_id": "2hop__1_1"}], header=["question_id"]))


def test_frozen_mapping_keeps_membership_when_collection_order_changes_and_rebuilds_indices():
    full = train_rows(30) + [row(f"2hop__v{k}_{k}", i=k, source="validation") for k in range(5)]
    first = manifest_for(full)
    rerun = recollected(full, seed=3)
    second = build_partition_manifest(rerun, frozen=frozen_from(first))
    assert {q: m["partition"] for q, m in second.items()} == {q: m["partition"] for q, m in first.items()}
    # Indices follow the new run, so each question still points at its own activation row...
    assert {q: m["full_index"] for q, m in second.items()} == {r["question_id"]: r["attempt2_activation_index"]
                                                              for r in rerun}
    # ...and they really did move, so this is not the first manifest in disguise.
    assert any(second[q]["full_index"] != first[q]["full_index"] for q in first)
    assert list(second["2hop__0_0"]) == MANIFEST_FIELDS
    assert partition_coverage(second, frozen_from(first))["missing"] == 0


def test_missing_question_fails_a_complete_run_and_never_moves_the_others():
    full = train_rows(200) + [row(f"3hop1__{k}_{k}", i=200 + k) for k in range(40)]
    first = manifest_for(full)
    frozen = frozen_from(first)
    gone = full[5]["question_id"]
    dropped = [r for r in full if r["question_id"] != gone]
    with pytest.raises(ValueError, match="were not collected"):
        build_partition_manifest(dropped, frozen=frozen)
    kept = build_partition_manifest(dropped, frozen=frozen, coverage="subset")
    assert all(kept[q]["partition"] == first[q]["partition"] for q in kept)
    assert set(first) - set(kept) == {gone}
    assert partition_coverage(kept, frozen) == {"expected": 240, "collected": 239, "missing": 1,
                                                "missing_ids": [gone]}
    # Negative control: re-deriving after the same drop re-cuts the set and moves other questions.
    rederived = manifest_for(dropped)
    assert any(rederived[q]["partition"] != first[q]["partition"] for q in rederived)


def test_subset_run_keeps_its_questions_partitions():
    full = train_rows(60) + [row(f"2hop__v{k}_{k}", i=k, source="validation") for k in range(10)]
    first = manifest_for(full)
    scan = recollected(full[::3], seed=5)                 # a deliberate subset, re-indexed
    subset = build_partition_manifest(scan, frozen=frozen_from(first), coverage="subset")
    assert all(subset[q]["partition"] == first[q]["partition"] for q in subset)
    coverage = partition_coverage(subset, frozen_from(first))
    assert coverage["collected"] == len(scan) and coverage["expected"] == len(full)
    assert coverage["missing"] == len(full) - len(scan) == len(coverage["missing_ids"])


def test_unknown_question_bad_source_split_and_mixed_arguments_are_refused():
    full = train_rows(10)
    frozen = frozen_from(manifest_for(full))
    with pytest.raises(ValueError, match="not in the frozen mapping"):
        build_partition_manifest(full + [row("2hop__99_99", i=10)], frozen=frozen)
    with pytest.raises(ValueError, match="source_split is 'validation'"):
        build_partition_manifest([{**full[0], "source_split": "validation"}] + full[1:], frozen=frozen)
    with pytest.raises(ValueError, match="pass one or the other"):
        build_partition_manifest(full, frozen=frozen, proportions=PROPORTIONS, seed=0)
    with pytest.raises(ValueError, match="required when no frozen"):
        build_partition_manifest(full)
    with pytest.raises(ValueError, match="coverage must be"):
        build_partition_manifest(full, frozen=frozen, coverage="partial")


def test_package_built_under_another_cut_is_caught_before_export():
    full = train_rows(50)
    first = manifest_for(full)
    frozen = frozen_from(first)
    assert check_manifest_matches_frozen(first, frozen) is first
    other = manifest_for(full, seed=1)
    assert other != first
    with pytest.raises(ValueError, match="partition is"):
        check_manifest_matches_frozen(other, frozen)
    with_stranger = {**first, "2hop__99_99": {**first["2hop__0_0"], "question_id": "2hop__99_99"}}
    with pytest.raises(ValueError, match="not in the frozen mapping"):
        check_manifest_matches_frozen(with_stranger, frozen)


# ------------------------------------------------------------------ the committed mapping

COMMITTED = (Path(__file__).resolve().parents[3] / "results" / "behavior"
             / "behavior_v011_split80_10_10_20261006")


@pytest.mark.skipif(not COMMITTED.is_dir(), reason="the committed behavior package is not in this checkout")
def test_committed_partitions_file_is_derived_from_the_committed_manifest():
    manifest = read_manifest(COMMITTED / "split_manifest.csv")
    frozen = read_partitions(COMMITTED / PARTITIONS_FILE)
    assert frozen == frozen_from(manifest)
    counts = json.loads((COMMITTED / "counts.json").read_text())
    assert counts["split_audit"]["by_partition"] == Counter(m["partition"] for m in manifest.values())


@pytest.mark.skipif(not COMMITTED.is_dir(), reason="the committed behavior package is not in this checkout")
def test_committed_manifest_is_reproduced_by_its_recorded_rule_from_the_same_question_set():
    manifest = read_manifest(COMMITTED / "split_manifest.csv")
    rule = json.loads((COMMITTED / "counts.json").read_text())["partition"]
    ids = {}
    for m in manifest.values():
        ids.setdefault(m["source_split"], []).append(m["question_id"])
    assignment = assign_partitions(ids, proportions=rule["proportions_of_source_train"], seed=rule["seed"])
    assert assignment == {q: m["partition"] for q, m in manifest.items()}
