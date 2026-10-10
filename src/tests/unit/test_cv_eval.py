"""Tests for mas_sae.probe.cv_eval."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from mas_sae.probe.cv_eval import (
    balanced_accuracy, bootstrap_ci, confusion, grouped_bootstrap_indices,
    make_group_folds, paired_bootstrap_diff, per_class_report,
)


def _toy():
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b", "c"], size=300, p=[0.7, 0.2, 0.1])
    pred = np.where(rng.random(300) < 0.6, y, rng.choice(["a", "b", "c"], size=300))
    return y, pred


def test_balanced_accuracy_matches_sklearn():
    y, pred = _toy()
    assert abs(balanced_accuracy(y, pred) - balanced_accuracy_score(y, pred)) < 1e-12


def test_balanced_accuracy_survives_missing_class_in_resample():
    assert balanced_accuracy(np.array(["a", "a"]), np.array(["a", "b"])) == 0.5


def test_confusion_and_per_class_report_consistent():
    y, pred = _toy()
    classes = ["a", "b", "c"]
    cm = confusion(y, pred, classes)
    assert cm.sum() == len(y)
    rep = per_class_report(y, pred, classes)
    for i, c in enumerate(classes):
        assert rep[c]["support"] == cm[i].sum()
        assert abs(rep[c]["recall"] - cm[i, i] / cm[i].sum()) < 1e-12


def test_group_folds_are_disjoint_cover_everything_and_reproducible():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(120), 2)                 # 2 rows per question
    y = np.repeat(rng.choice(["a", "b", "c"], size=120, p=[0.6, 0.25, 0.15]), 2)
    folds, fold_id = make_group_folds(y, groups, 5, seed=3)
    seen = np.concatenate([test for _, test in folds])
    assert sorted(seen.tolist()) == list(range(len(y)))   # every row tested exactly once
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test]) # no question on both sides
    folds2, fold_id2 = make_group_folds(y, groups, 5, seed=3)
    assert np.array_equal(fold_id, fold_id2)


def test_bootstrap_ci_brackets_point_estimate_and_paired_diff_has_right_sign():
    y, pred = _toy()
    groups = np.arange(len(y))
    boot = grouped_bootstrap_indices(groups, 400, seed=0)
    lo, hi = bootstrap_ci(y, pred, boot)
    assert lo <= balanced_accuracy(y, pred) <= hi
    worse = np.where(np.random.default_rng(5).random(len(y)) < 0.5, pred, "a")
    diff, dlo, dhi = paired_bootstrap_diff(y, pred, worse, boot)
    assert diff > 0 and dlo <= diff <= dhi


def test_grouped_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(10), 3)
    for idx in grouped_bootstrap_indices(groups, 20, seed=0):
        counts = np.bincount(groups[idx], minlength=10)
        assert set(counts % 3) == {0}                      # groups enter in complete blocks of 3
