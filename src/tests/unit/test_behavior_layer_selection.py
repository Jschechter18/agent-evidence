"""Alignment, partition discipline and layer selection on a fake shared-Drive tree.

Layer 17 carries a planted signal; layer 8 is noise. The pipeline must find that, must never read
test/intervention rows, must verify the manifest index against the interactions file, and must give
NO recommendation on a tie or when nothing beats chance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mas_sae.probe import behavior_data as bd
from mas_sae.probe import layer_selection as ls

CLASSES = ["adopted_critic", "retained_a1", "third_answer"]   # solver_response values
CODE = {"adopted_critic": 0, "retained_a1": 1, "third_answer": 1}      # planted signal: dim 0 = adopted, dim 1 = not


def build_tree(root: Path, n_train=700, n_val=220, seed=0, labels_have_episode_id=False,
               corrupt_index=False, swap_source_split=False, old_manifest=False):
    """Fake Drive: results/{train,validation}/interactions.jsonl, acts/layer_NN/*_attempt2.pt,
    labels.csv and a four-partition manifest."""
    rng = np.random.default_rng(seed)
    q, rows = 0, []
    for split, n in (("train", n_train), ("validation", n_val)):
        for i in range(n):
            partition = ("validation" if split == "validation"
                         else str(rng.choice(["train", "test", "intervention"], p=[0.8, 0.1, 0.1])))
            rows.append(dict(question_id=f"q{q:05d}", source_split=split, row=i, partition=partition,
                             target=str(rng.choice(CLASSES, p=[0.6, 0.2, 0.2])),   # = solver_response
                             eligible=bool(rng.random() < 0.9)))
            q += 1
    truth = pd.DataFrame(rows)

    for split in ("train", "validation"):
        sub = truth[truth.source_split == split]
        d = root / "results" / split
        d.mkdir(parents=True)
        with open(d / "interactions.jsonl", "w") as f:
            for r in sub.itertuples():
                f.write(json.dumps({
                    "episode_id": f"{r.question_id}__natural", "question_id": r.question_id,
                    "attempt2_activation_index": int(r.row),
                    "solver_attempt_1": f"a1 {'alpha' if r.target == 'adopted_critic' else 'beta'} {rng.integers(1000)}",
                    "critic_advocated_answer": "x", "critic_feedback": "fb"}) + "\n")

    for layer, signal in ((8, 0.0), (17, 2.0)):
        d = root / "acts" / f"layer_{layer:02d}"
        d.mkdir(parents=True)
        for split in ("train", "validation"):
            sub = truth[truth.source_split == split]
            X = rng.normal(size=(len(sub), 16)).astype(np.float32)
            for i, t in enumerate(sub.target):
                X[i, CODE[t]] += signal
            torch.save(torch.from_numpy(X), d / f"{split}_attempt2.pt")

    # real labels.csv: primary_target is numeric (1.0 adopted / 0.0 not / blank if not eligible);
    # it has question_id but NO episode_id, and carries its own source_split/partition columns.
    lab = pd.DataFrame({"question_id": truth.question_id, "source_split": truth.source_split,
                        "partition": truth.partition, "solver_response": truth.target,
                        "eligible_primary": truth.eligible,
                        "primary_target": np.where(truth.eligible, (truth.target == "adopted_critic").astype(float), np.nan)})
    if labels_have_episode_id:
        lab.insert(0, "episode_id", truth.question_id + "__natural")
    lab.to_csv(root / "labels.csv", index=False)

    index = truth.row.to_numpy().copy()
    if corrupt_index:
        index = np.roll(index, 5)
    man = pd.DataFrame({"question_id": truth.question_id, "source_split": truth.source_split,
                        "partition": truth.partition, "hop_group": "2hop",
                        "full_index": index, "scan_index": index})
    if swap_source_split:
        man.loc[:4, "source_split"] = np.where(man.loc[:4, "source_split"] == "train", "validation", "train")
    if old_manifest:
        man = man.rename(columns={"partition": "canonical_split"}).drop(columns=["source_split"])
    man.to_csv(root / "manifest.csv", index=False)
    return truth


def make_cfg(root: Path, **kw):
    base = dict(capstone_data_root=root, activations_dir="acts", results_dir="results",
                labels_csv="labels.csv", manifest_csv="manifest.csv", index_column="full_index",
                layers=[8, 17], n_bootstrap=100, output_dir=root / "out")
    base.update(kw)
    return ls.LayerSelectionConfig(**base)


def _summary(run_dir):
    return json.loads((run_dir / "layer_selection_summary.json").read_text())


# ------------------------------------------------------------------ partition discipline

def test_sealed_partitions_are_never_selected_and_cannot_be_requested(tmp_path):
    truth = build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    episodes = bd.build_episode_table(tmp_path / "results", ["train", "validation"], [])
    aligned, _ = bd.align_episodes(episodes, pd.read_csv(tmp_path / "labels.csv"),
                                   pd.read_csv(tmp_path / "manifest.csv"), bd.Columns(), "full_index")
    chosen, _ = bd.select_partitions(aligned, bd.Columns(), ["train", "validation"])
    assert set(chosen["partition"]) == {"train", "validation"}
    ok = truth[truth.partition.isin(["train", "validation"]) & truth.eligible]
    assert set(chosen["ep_question_id"]) == set(ok.question_id)
    for sealed in ("test", "intervention"):
        try:
            bd.select_partitions(aligned, bd.Columns(), [sealed])
        except PermissionError as e:
            assert "sealed" in str(e)
        else:
            raise AssertionError(f"{sealed} was released during development")


def test_holdout_fits_on_train_only_and_scores_on_validation_only(tmp_path):
    truth = build_tree(tmp_path)
    run_dir = ls.run(make_cfg(tmp_path))
    comp = _summary(run_dir)["comparison"]
    el = truth[truth.eligible]
    assert comp["n_rows_fit"] == int(((el.partition == "train")).sum())
    assert comp["n_rows_evaluated"] == int(((el.partition == "validation")).sum())
    used = pd.read_csv(run_dir / "rows_and_folds.csv")
    assert set(used["partition"]) == {"train", "validation"}


# ------------------------------------------------------------------------- alignment

def test_labels_stay_attached_to_the_right_activation_rows(tmp_path):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    selected, _, cols = ls.load_inputs(cfg)
    X = bd.load_layer_matrix(selected, tmp_path / "acts" / "layer_17")
    d = [0 if t == "adopted_critic" else 1 for t in selected["target_name"]]
    own = np.array([X[i, k] for i, k in enumerate(d)])
    other = np.array([X[i, 1 - k] for i, k in enumerate(d)])
    assert own.mean() - other.mean() > 1.5             # planted +2 sits on exactly the labelled class


def test_labels_join_uses_question_id_as_the_real_labels_file_has_no_episode_id(tmp_path):
    build_tree(tmp_path)
    _, diag, _ = ls.load_inputs(make_cfg(tmp_path))
    assert diag["labels_join"] == "question_id"


def test_labels_join_prefers_episode_id_when_present(tmp_path):
    build_tree(tmp_path, labels_have_episode_id=True)
    _, diag, _ = ls.load_inputs(make_cfg(tmp_path))
    assert diag["labels_join"] == "episode_id"


def test_numeric_target_gets_readable_names_and_unexpected_values_are_refused():
    assert list(bd.target_names(pd.Series([1.0, 0.0, 1.0]))) == ["adopted_critic", "not_adopted", "adopted_critic"]
    try:
        bd.target_names(pd.Series([1.0, 2.0]))
    except bd.AlignmentError as e:
        assert "unexpected" in str(e)
    else:
        raise AssertionError("an unexpected target value was accepted")


def test_index_that_disagrees_with_interactions_is_rejected(tmp_path):
    build_tree(tmp_path, corrupt_index=True)
    try:
        ls.load_inputs(make_cfg(tmp_path))
    except bd.AlignmentError as e:
        assert "disagrees" in str(e)
    else:
        raise AssertionError("a misaligned manifest index was accepted")


def test_source_split_mismatch_is_rejected(tmp_path):
    build_tree(tmp_path, swap_source_split=True)
    try:
        ls.load_inputs(make_cfg(tmp_path))
    except bd.AlignmentError as e:
        assert "source_split" in str(e)
    else:
        raise AssertionError("a source_split mismatch was accepted")


def test_old_canonical_split_manifest_is_rejected_with_a_clear_message(tmp_path):
    build_tree(tmp_path, old_manifest=True)
    try:
        ls.load_inputs(make_cfg(tmp_path))
    except bd.AlignmentError as e:
        assert "NEW" in str(e) and "partition" in str(e)
    else:
        raise AssertionError("the old manifest was accepted")


def test_misnamed_text_field_fails_loudly_instead_of_emptying_the_baseline(tmp_path):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path, text_fields=["solver_attempt_1", "no_such_field"])
    try:
        ls.load_inputs(cfg)
    except bd.AlignmentError as e:
        assert "no_such_field" in str(e)
    else:
        raise AssertionError("an all-empty text column was accepted")


# ------------------------------------------------------------------------ the protocols

def test_holdout_finds_planted_layer_and_beats_noise_layer(tmp_path):
    build_tree(tmp_path)
    comp = _summary(ls.run(make_cfg(tmp_path)))["comparison"]
    assert comp["protocol"] == "holdout" and comp["recommended_layer"] == "layer_17"
    r17, r08 = comp["results"]["layer_17"], comp["results"]["layer_08"]
    assert r17["balanced_accuracy"] > r08["balanced_accuracy"] + 0.15
    assert comp["chance_balanced_accuracy"] == 0.5 and comp["classes"] == ["adopted_critic", "not_adopted"]
    assert r17["balanced_accuracy"] > comp["chance_balanced_accuracy"] + 0.2
    assert comp["top_layer_ci_lower_above_chance"] is True
    by_resp = r17["recall_by_solver_response"]            # the 3-way breakdown stays visible
    assert set(by_resp) == set(CLASSES) and sum(v["n"] for v in by_resp.values()) == comp["n_rows_evaluated"]
    assert by_resp["retained_a1"]["recall"] > 0.5 and by_resp["third_answer"]["recall"] > 0.5
    assert "text_only_baseline" in comp["results"]


def test_cv_protocol_uses_only_development_partitions_and_finds_planted_layer(tmp_path):
    build_tree(tmp_path)
    run_dir = ls.run(make_cfg(tmp_path, protocol="cv"))
    comp = _summary(run_dir)["comparison"]
    assert comp["protocol"] == "cv" and comp["recommended_layer"] == "layer_17"
    used = pd.read_csv(run_dir / "rows_and_folds.csv")
    assert set(used["partition"]) == {"train", "validation"}      # test/intervention never touched
    assert used["question_id"].is_unique and set(used["fold"]) == set(range(5))
    assert set(used["solver_response"]) == set(CLASSES)


def _toy(seed=0, n=300, d=16, signal=0.0):
    rng = np.random.default_rng(seed)
    y = rng.choice(CLASSES, size=n, p=[0.5, 0.25, 0.25])
    X = rng.normal(size=(n, d)).astype(np.float32)
    for i, c in enumerate(y):
        X[i, CLASSES.index(c)] += signal
    return y, np.arange(n).astype(str), X


def test_exact_tie_gives_no_recommendation_cv_and_holdout():
    y, groups, X = _toy(signal=2.0)
    cfg = ls.LayerSelectionConfig(n_bootstrap=50, c_grid=[0.1, 1.0])
    two = {"layer_a": X, "layer_b": X.copy()}
    cv = ls.compare_layers(two, y, groups, None, cfg)
    is_fit = np.arange(len(y)) < 200
    ho = ls.compare_layers_holdout(two, y, groups, is_fit, None, cfg)
    for res in (cv, ho):
        assert res["recommended_layer"] is None and res["no_recommendation_reason"] == "tie"
        assert len(res["tied_with_top"]) == 1


def test_no_recommendation_when_nothing_beats_chance():
    y, groups, X1 = _toy(seed=1)
    _, _, X2 = _toy(seed=2)                       # two pure-noise layers
    cfg = ls.LayerSelectionConfig(n_bootstrap=100, c_grid=[0.1, 1.0])
    res = ls.compare_layers({"layer_a": X1, "layer_b": X2}, y, groups, None, cfg)
    assert res["recommended_layer"] is None
    assert res["no_recommendation_reason"] in {"tie", "top_layer_not_distinguishable_from_chance"}


def test_tie_tolerance_widens_what_counts_as_a_tie():
    y, groups, X_strong = _toy(signal=2.0)
    _, _, X_weak = _toy(seed=3, signal=1.2)
    base = dict(n_bootstrap=50, c_grid=[0.1, 1.0])
    clear = ls.compare_layers({"a": X_strong, "b": X_weak}, y, groups, None, ls.LayerSelectionConfig(**base))
    wide = ls.compare_layers({"a": X_strong, "b": X_weak}, y, groups, None,
                             ls.LayerSelectionConfig(tie_tolerance=1.0, **base))
    assert clear["recommended_layer"] is not None and wide["recommended_layer"] is None


def test_label_shuffle_control_lands_near_chance_for_both_protocols(tmp_path):
    build_tree(tmp_path)
    for protocol in ("holdout", "cv"):
        comp = _summary(ls.run(make_cfg(tmp_path, protocol=protocol, n_permutations=8)))["comparison"]
        ctrl = comp["label_shuffle_control"]["layer_17"]
        assert abs(ctrl["null_mean"] - 0.5) < 0.1
        assert ctrl["observed"] > ctrl["null_p95"]


def test_exported_layout_is_verified_through_the_original_row_index():
    """Exported partition files re-index rows (attempt2_activation_index) and keep the original row in
    source_activation_index; the manifest's full_index refers to the ORIGINAL row."""
    ep = pd.DataFrame({"source_split": ["train", "train"], "activation_row": [0, 1], "verify_index": [10, 11],
                       "ep_episode_id": ["a__natural", "b__natural"], "ep_question_id": ["a", "b"]})
    lab = pd.DataFrame({"episode_id": ["a__natural", "b__natural"], "primary_target": ["adopted_critic"] * 2,
                        "eligible_primary": [True, True]})
    man = pd.DataFrame({"question_id": ["a", "b"], "source_split": ["train", "train"],
                        "partition": ["train", "train"], "full_index": [10, 11]})
    df, diag = bd.align_episodes(ep, lab, man, bd.Columns(), "full_index")
    assert diag["index_verified_rows"] == 2 and list(df["activation_row"]) == [0, 1]   # tensors use the new rows
    man_bad = man.assign(full_index=[10, 99])
    try:
        bd.align_episodes(ep, lab, man_bad, bd.Columns(), "full_index")
    except bd.AlignmentError as e:
        assert "disagrees" in str(e)
    else:
        raise AssertionError("a wrong original index was accepted for exported data")


def test_stale_labels_path_is_refused_until_both_files_are_set(tmp_path):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path, labels_csv="REPLACE_ME/labels.csv")
    try:
        ls.load_inputs(cfg)
    except SystemExit as e:
        assert "labels_csv" in str(e)
    else:
        raise AssertionError("ran with a placeholder labels path")
