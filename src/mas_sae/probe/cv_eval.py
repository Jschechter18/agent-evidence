"""Question-grouped cross-validation and uncertainty-aware metrics.

Everything here is shared by layer selection and the SAE-feature probe so that
every comparison uses the *same* folds and the *same* metric definitions.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean per-class recall over the classes present in ``y_true``.

    Equals sklearn's balanced_accuracy_score when every class is present, but stays
    defined (and quiet) on bootstrap resamples that happen to miss a rare class.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def macro_f1(y_true, y_pred, classes) -> float:
    return float(np.mean([per_class_report(y_true, y_pred, classes)[str(c)]["f1"]
                          for c in classes]))


def confusion(y_true, y_pred, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def per_class_report(y_true, y_pred, classes) -> dict:
    cm = confusion(y_true, y_pred, classes)
    report = {}
    for i, c in enumerate(classes):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[str(c)] = {"precision": float(precision), "recall": float(recall),
                          "f1": float(f1), "support": int(support)}
    return report


def make_group_folds(y, groups, n_splits: int, seed: int):
    """Stratified, group-disjoint folds. Returns (list of (train, test), fold_id array).

    No group (question) ever appears on both sides of a fold. Build this ONCE and
    reuse it for every layer so layers are compared on identical folds.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    smallest = min(counts.values())
    if smallest < n_splits:
        warnings.warn(
            f"Rarest class has only {smallest} rows for {n_splits} folds ({counts}); some "
            "folds will contain none of it and per-class metrics will be very noisy.",
            stacklevel=2,
        )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn's own rare-class warning; we warned above
        folds = list(cv.split(np.zeros(len(y)), y, groups))
    fold_id = np.empty(len(y), dtype=int)
    for k, (_, test) in enumerate(folds):
        fold_id[test] = k
    for train, test in folds:  # hard guarantee, not an assumption
        assert not set(groups[train]) & set(groups[test]), "group leaked across a fold"
    return folds, fold_id


def grouped_bootstrap_indices(groups, n_boot: int, seed: int) -> list[np.ndarray]:
    """Resample whole groups (questions) with replacement. Reuse the same list for
    every model so differences between models are *paired*."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        for _ in range(n_boot)
    ]


def bootstrap_ci(y_true, y_pred, boot_indices, alpha: float = 0.05, metric=balanced_accuracy):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = np.array([metric(y_true[i], y_pred[i]) for i in boot_indices])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y_true, pred_a, pred_b, boot_indices, alpha: float = 0.05,
                          metric=balanced_accuracy):
    """metric(A) - metric(B) with a paired bootstrap CI (same resamples for both)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    diffs = np.array([metric(y_true[i], pred_a[i]) - metric(y_true[i], pred_b[i])
                      for i in boot_indices])
    point = metric(y_true, pred_a) - metric(y_true, pred_b)
    return float(point), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))
