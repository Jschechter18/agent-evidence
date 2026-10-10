"""Layer selection for the Solver-Critic activation probe, under the 6 Oct split contract.

Two protocols, both restricted to the DEVELOPMENT partitions (``test`` and ``intervention`` are
sealed and can never be requested):

  holdout  fit the probe on `train`, rank layers on `validation`   (the contract; use for the
           full-run layers 8/17/25/33, where validation has hundreds of eligible rows)
  cv       question-grouped CV over `train` + `validation`          (use for the nine-layer scan,
           whose validation slice alone is far too small to rank nine layers)

Common to both: the target is `primary_target` on `eligible_primary` rows, the probe is
class-weighted with C chosen inside the training data only, every layer sees identical rows/folds,
metrics are balanced accuracy + per-class precision/recall + confusion matrix + bootstrap CIs, a
text-only baseline (A1 + Critic text) is scored the same way, and ties (or nothing beating chance)
produce NO recommendation.

Entry point: ``scripts/layer_selection.py``.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mas_sae.probe import behavior_data as bd
from mas_sae.probe.cv_eval import (
    balanced_accuracy, bootstrap_ci, confusion, grouped_bootstrap_indices,
    macro_f1, make_group_folds, paired_bootstrap_diff, per_class_report,
)
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)

PATH_FIELDS = {"capstone_data_root", "output_dir"}
DEV_PARTITIONS = ["train", "validation"]


@dataclass
class LayerSelectionConfig:
    capstone_data_root: Path = Path("data/capstone-data")  # the downloaded shared Drive folder
    activations_dir: str = "natural_4b_full/data"          # <dir>/layer_NN/{split}_attempt2.pt
    results_dir: str = "natural_4b_full/results"           # <dir>/{split}/interactions.jsonl
    # labels.csv and split_manifest.csv are written together into ONE behavior package folder;
    # point both at the NEW package (the one whose manifest has a `partition` column).
    labels_csv: str = "REPLACE_ME/labels.csv"
    manifest_csv: str = "REPLACE_ME/split_manifest.csv"
    index_column: str = "full_index"                       # "scan_index" for the nine-layer scan
    layers: list = field(default_factory=lambda: [8, 17, 25, 33])
    source_splits: list = field(default_factory=lambda: ["train", "validation"])  # folder names
    activation_file: str = "{split}_attempt2.pt"           # Solver activations before A2
    # primary_target is binary (1 = adopted the Critic, 0 = retained A1 or third answer); solver_response
    # carries the 3-way breakdown and is reported per layer. For a sensitivity run on adopted-vs-retained
    # only, set target_column: strict_target and eligible_column: eligible_strict.
    target_column: str = "primary_target"
    eligible_column: str = "eligible_primary"
    protocol: str = "holdout"                              # "holdout" | "cv"
    fit_partition: str = "train"
    eval_partition: str = "validation"
    # text-only baseline: everything the Solver can already SEE in the A2 prompt.
    # Never add solver_attempt_2 or any label-derived field here (that would leak the answer).
    text_fields: list = field(default_factory=lambda: [
        "solver_attempt_1", "critic_advocated_answer", "critic_feedback"])
    n_folds: int = 5                                       # cv protocol only
    seed: int = 42
    # C is chosen INSIDE the training data (inner question-grouped CV on balanced accuracy), never on
    # the evaluation rows. With thousands of features a single fixed C can collapse every layer to a
    # constant prediction, so the grid is wide.
    c_grid: list = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    inner_folds: int = 3
    n_bootstrap: int = 1000
    # Layers whose balanced accuracy differs from the top by <= this are TIED -> no recommendation.
    tie_tolerance: float = 1e-9
    n_permutations: int = 0  # >0: label-shuffle control on the top layer only
    output_dir: Path = Path("results/layer_selection")

    @classmethod
    def from_yaml(cls, path: Path) -> "LayerSelectionConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for k in PATH_FIELDS & set(raw):
            raw[k] = Path(raw[k])
        cfg = cls(**raw)
        if cfg.protocol not in {"holdout", "cv"}:
            raise ValueError("protocol must be 'holdout' or 'cv'")
        return cfg


# ----------------------------------------------------------------------------- models

def _activation_model(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def _text_model(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def _fit_with_inner_cv(make_model, X, y, groups, c_grid, inner_folds, seed):
    """Pick C by inner grouped CV on THIS training set only, then refit on all of it.
    Ties go to the smaller C (stronger regularization)."""
    n_inner = int(max(2, min(inner_folds, min(np.unique(y, return_counts=True)[1]))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inner, _ = make_group_folds(y, groups, n_inner, seed)
    best_c, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for tr, va in inner:
            m = make_model(C).fit(X[tr], y[tr])
            scores.append(balanced_accuracy(y[va], m.predict(X[va])))
        if np.mean(scores) > best_score + 1e-9:
            best_c, best_score = C, float(np.mean(scores))
    return make_model(best_c).fit(X, y), best_c


def _fit_counting_warnings(make_model, X, y, groups, cfg):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, C = _fit_with_inner_cv(make_model, X, y, groups, cfg.c_grid, cfg.inner_folds, cfg.seed)
    return model, C, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def oof_predict(make_model, X, y, groups, folds, c_grid, inner_folds, seed):
    """Out-of-fold predictions. Scaler/vectorizer AND C are refit inside every outer fold."""
    y, groups = np.asarray(y), np.asarray(groups)
    pred = np.empty(len(y), dtype=y.dtype)
    n_conv, chosen = 0, []
    for train, test in folds:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, C = _fit_with_inner_cv(make_model, X[train], y[train], groups[train],
                                          c_grid, inner_folds, seed)
        n_conv += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        chosen.append(C)
        pred[test] = model.predict(X[test])
    return pred, n_conv, chosen


# ------------------------------------------------------------------------- comparison

def _candidates(layer_matrices, texts, cfg):
    act = lambda C: _activation_model(C, cfg.seed)
    cand = {name: (X, act) for name, X in layer_matrices.items()}
    if texts is not None:
        cand["text_only_baseline"] = (np.asarray(texts, dtype=object),
                                      lambda C: _text_model(C, cfg.seed))
    return cand


def _evaluate(preds, extras, y, groups, classes, cfg, folds=None, subgroups=None) -> dict:
    """Metrics, CIs, paired differences, tie rule. ``y``/``groups`` are the EVALUATED rows."""
    boot = grouped_bootstrap_indices(groups, cfg.n_bootstrap, cfg.seed)
    records = {}
    for name, pred in preds.items():
        lo, hi = bootstrap_ci(y, pred, boot)
        rec = {
            "balanced_accuracy": balanced_accuracy(y, pred), "ci95": [lo, hi],
            "macro_f1": macro_f1(y, pred, classes),
            "per_class": per_class_report(y, pred, classes),
            "confusion_matrix": {"labels": [str(c) for c in classes],
                                 "matrix": confusion(y, pred, classes).tolist()},
            **extras[name],
        }
        if subgroups is not None:   # e.g. solver_response: did the probe get retained_a1 / third_answer rows right?
            rec["recall_by_solver_response"] = {
                str(g): {"recall": float(np.mean(pred[subgroups == g] == y[subgroups == g])),
                         "n": int((subgroups == g).sum())} for g in np.unique(subgroups)}
        if folds is not None:
            fs = [balanced_accuracy(y[test], pred[test]) for _, test in folds]
            rec.update(per_fold_balanced_accuracy=fs, per_fold_mean=float(np.mean(fs)),
                       per_fold_sd=float(np.std(fs, ddof=1)))
        records[name] = rec

    act_names = [n for n in preds if n != "text_only_baseline"]
    best = max(act_names, key=lambda n: records[n]["balanced_accuracy"])
    for name in preds:
        d, dlo, dhi = paired_bootstrap_diff(y, preds[name], preds[best], boot)
        records[name]["diff_vs_best"] = {"diff": d, "ci95": [dlo, dhi]}
        records[name]["within_noise_of_best"] = bool(dlo <= 0 <= dhi)
    top_ba = records[best]["balanced_accuracy"]
    tied = [n for n in act_names
            if n != best and abs(records[n]["balanced_accuracy"] - top_ba) <= cfg.tie_tolerance]
    above_chance = bool(records[best]["ci95"][0] > 1.0 / len(classes))
    reason = "tie" if tied else (None if above_chance else "top_layer_not_distinguishable_from_chance")
    beats_text = best_vs_text = None
    if "text_only_baseline" in preds:
        d, dlo, dhi = paired_bootstrap_diff(y, preds[best], preds["text_only_baseline"], boot)
        beats_text, best_vs_text = bool(dlo > 0), {"diff": d, "ci95": [dlo, dhi]}
    return {
        "classes": [str(c) for c in classes],
        "n_rows_evaluated": int(len(y)),
        "class_counts_evaluated": {str(c): int((y == c).sum()) for c in classes},
        "chance_balanced_accuracy": 1.0 / len(classes),
        "c_grid": list(cfg.c_grid),
        "recommended_layer": None if reason else best,   # None on a tie or when nothing beats chance
        "no_recommendation_reason": reason,
        "top_layer_by_balanced_accuracy": best,
        "tied_with_top": tied,
        "top_layer_ci_lower_above_chance": above_chance,
        "top_layer_beats_text_baseline": beats_text,
        "top_layer_vs_text_baseline": best_vs_text,
        "layers_indistinguishable_from_best": [n for n in act_names if records[n]["within_noise_of_best"]],
        "results": records,
    }


def compare_layers(layer_matrices, y, groups, texts, cfg, subgroups=None) -> dict:
    """`cv` protocol: question-grouped CV over the given rows (all from development partitions)."""
    y, groups = np.asarray(y), np.asarray(groups)
    subgroups = None if subgroups is None else np.asarray(subgroups).astype(str)
    classes = sorted(np.unique(y).tolist())
    folds, fold_id = make_group_folds(y, groups, cfg.n_folds, cfg.seed)  # ONE set of folds
    preds, extras = {}, {}
    for name, (X, factory) in _candidates(layer_matrices, texts, cfg).items():
        pred, n_conv, chosen = oof_predict(factory, X, y, groups, folds, cfg.c_grid, cfg.inner_folds, cfg.seed)
        preds[name], extras[name] = pred, {"chosen_C_per_fold": chosen, "n_convergence_warnings": n_conv}
    out = _evaluate(preds, extras, y, groups, classes, cfg, folds, subgroups)
    out.update(protocol="cv", n_groups=int(len(np.unique(groups))), _fold_id=fold_id.tolist())
    if cfg.n_permutations > 0:
        best = out["top_layer_by_balanced_accuracy"]
        out["label_shuffle_control"] = {best: _permutation_cv(
            layer_matrices[best], y, groups, folds, cfg, preds[best])}
    return out


def compare_layers_holdout(layer_matrices, y, groups, is_fit, texts, cfg, subgroups=None) -> dict:
    """`holdout` protocol: fit on the rows where ``is_fit`` is True, score on all the others."""
    y, groups, is_fit = np.asarray(y), np.asarray(groups), np.asarray(is_fit, dtype=bool)
    fit_idx, eval_idx = np.flatnonzero(is_fit), np.flatnonzero(~is_fit)
    subgroups = None if subgroups is None else np.asarray(subgroups).astype(str)[eval_idx]
    if len(fit_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("holdout needs rows in both the fit and the evaluation partition")
    assert not set(groups[fit_idx]) & set(groups[eval_idx]), "a question appears in both partitions"
    classes = sorted(np.unique(y).tolist())
    if len(np.unique(y[eval_idx])) < 2:
        raise ValueError("the evaluation partition contains a single class; metrics are undefined")
    preds, extras = {}, {}
    for name, (X, factory) in _candidates(layer_matrices, texts, cfg).items():
        model, C, n_conv = _fit_counting_warnings(factory, X[fit_idx], y[fit_idx], groups[fit_idx], cfg)
        preds[name] = model.predict(X[eval_idx])
        extras[name] = {"chosen_C": C, "n_convergence_warnings": n_conv}
    out = _evaluate(preds, extras, y[eval_idx], groups[eval_idx], classes, cfg, None, subgroups)
    out.update(protocol="holdout", n_rows_fit=int(len(fit_idx)),
               class_counts_fit={str(c): int((y[fit_idx] == c).sum()) for c in classes})
    if cfg.n_permutations > 0:
        best = out["top_layer_by_balanced_accuracy"]
        out["label_shuffle_control"] = {best: _permutation_holdout(
            layer_matrices[best], y, groups, fit_idx, eval_idx, cfg, preds[best])}
    return out


def _shuffle_labels(y, groups, rng):
    """Permute labels across questions (whole groups), preserving the class balance."""
    uniq = np.unique(groups)
    first = {g: y[np.flatnonzero(groups == g)[0]] for g in uniq}
    labels = np.array([first[g] for g in uniq])
    lookup = {g: i for i, g in enumerate(uniq)}
    return rng.permutation(labels)[np.array([lookup[g] for g in groups])]


def _summarise_null(observed, null) -> dict:
    null = np.array(null)
    return {"observed": float(observed), "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)), "n_permutations": int(len(null)),
            "p_value": float((1 + (null >= observed).sum()) / (1 + len(null)))}


def _permutation_cv(X, y, groups, folds, cfg, observed_pred) -> dict:
    rng = np.random.default_rng(cfg.seed + 1)
    model = lambda C: _activation_model(C, cfg.seed)
    null = []
    for _ in range(cfg.n_permutations):
        y_perm = _shuffle_labels(y, groups, rng)
        pred, _, _ = oof_predict(model, X, y_perm, groups, folds, cfg.c_grid, cfg.inner_folds, cfg.seed)
        null.append(balanced_accuracy(y_perm, pred))
    return _summarise_null(balanced_accuracy(y, observed_pred), null)


def _permutation_holdout(X, y, groups, fit_idx, eval_idx, cfg, observed_pred) -> dict:
    """Shuffle the TRAINING labels, refit the whole tuned procedure, score on the real eval labels."""
    rng = np.random.default_rng(cfg.seed + 1)
    model = lambda C: _activation_model(C, cfg.seed)
    null = []
    for _ in range(cfg.n_permutations):
        y_fit = _shuffle_labels(y[fit_idx], groups[fit_idx], rng)
        m, _ = _fit_with_inner_cv(model, X[fit_idx], y_fit, groups[fit_idx], cfg.c_grid, cfg.inner_folds, cfg.seed)
        null.append(balanced_accuracy(y[eval_idx], m.predict(X[eval_idx])))
    return _summarise_null(balanced_accuracy(y[eval_idx], observed_pred), null)


# ------------------------------------------------------------------------------- I/O

def load_inputs(cfg: LayerSelectionConfig, cols: bd.Columns | None = None):
    """Read, align and filter everything except the activation tensors."""
    cols = cols or bd.Columns(target=cfg.target_column, eligible=cfg.eligible_column)
    if "REPLACE_ME" in str(cfg.manifest_csv) or "REPLACE_ME" in str(cfg.labels_csv):
        raise SystemExit("Set labels_csv AND manifest_csv in the config to the NEW behavior package folder "
                         "(they are written together; the manifest has a `partition` column).")
    root = Path(cfg.capstone_data_root)
    episodes = bd.build_episode_table(root / cfg.results_dir, cfg.source_splits, cfg.text_fields)
    labels = pd.read_csv(root / cfg.labels_csv)
    manifest = pd.read_csv(root / cfg.manifest_csv)
    aligned, align_diag = bd.align_episodes(episodes, labels, manifest, cols, cfg.index_column)
    parts = [cfg.fit_partition, cfg.eval_partition] if cfg.protocol == "holdout" else DEV_PARTITIONS
    selected, sel_diag = bd.select_partitions(aligned, cols, parts)
    return selected, {**align_diag, **sel_diag, "protocol": cfg.protocol}, cols


def run(cfg: LayerSelectionConfig, cols: bd.Columns | None = None) -> Path:
    selected, diag, cols = load_inputs(cfg, cols)
    y = selected["target_name"].to_numpy()
    response = selected[cols.response].astype(str).to_numpy() if cols.response in selected else None
    groups = selected["ep_question_id"].astype(str).to_numpy()
    texts = (selected[[f"text_{f}" for f in cfg.text_fields]]
             .agg(" || ".join, axis=1).to_numpy() if cfg.text_fields else None)

    root = Path(cfg.capstone_data_root)
    matrices = {}
    for layer in cfg.layers:
        name = f"layer_{int(layer):02d}"
        matrices[name] = bd.load_layer_matrix(selected, root / cfg.activations_dir / name,
                                              cfg.activation_file)
        logger.info("loaded %s: %s", name, matrices[name].shape)

    if cfg.protocol == "holdout":
        is_fit = (selected[cols.partition].astype(str) == cfg.fit_partition).to_numpy()
        res = compare_layers_holdout(matrices, y, groups, is_fit, texts, cfg, response)
    else:
        res = compare_layers(matrices, y, groups, texts, cfg, response)

    run_dir = create_versioned_run_dir(cfg.output_dir)
    fold_id = res.pop("_fold_id", None)
    rows = pd.DataFrame({"question_id": groups, "partition": selected[cols.partition].to_numpy(), "target": y})
    if response is not None:
        rows["solver_response"] = response
    if fold_id is not None:
        rows["fold"] = fold_id
    rows.to_csv(run_dir / "rows_and_folds.csv", index=False)   # the exact definition of what was used
    _write_table(res, run_dir / "layer_comparison.csv")
    (run_dir / "layer_selection_summary.json").write_text(json.dumps(
        {"data_diagnostics": diag, "config": json.loads(json.dumps(asdict(cfg), default=str)),
         "comparison": res}, indent=2, default=str))
    logger.info("wrote %s", run_dir)
    return run_dir


def _write_table(res: dict, path: Path) -> None:
    rows = []
    for name, r in res["results"].items():
        row = {"name": name, "balanced_accuracy": r["balanced_accuracy"],
               "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1], "macro_f1": r["macro_f1"],
               "diff_vs_best": r["diff_vs_best"]["diff"],
               "diff_ci95_lo": r["diff_vs_best"]["ci95"][0], "diff_ci95_hi": r["diff_vs_best"]["ci95"][1],
               "within_noise_of_best": r["within_noise_of_best"],
               "median_chosen_C": float(np.median(r.get("chosen_C_per_fold", [r.get("chosen_C", np.nan)])))}
        if "per_fold_mean" in r:
            row["per_fold_mean"], row["per_fold_sd"] = r["per_fold_mean"], r["per_fold_sd"]
        for c, m in r["per_class"].items():
            row[f"recall_{c}"], row[f"precision_{c}"], row[f"n_{c}"] = m["recall"], m["precision"], m["support"]
        for g, m in r.get("recall_by_solver_response", {}).items():
            row[f"resp_recall_{g}"], row[f"resp_n_{g}"] = m["recall"], m["n"]
        rows.append(row)
    pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).to_csv(path, index=False)


def inspect_inputs(cfg: LayerSelectionConfig, cols: bd.Columns | None = None) -> None:
    """Print what the files actually contain, and whether alignment verifies."""
    cols = cols or bd.Columns(target=cfg.target_column, eligible=cfg.eligible_column)
    root = Path(cfg.capstone_data_root)
    if "REPLACE_ME" in str(cfg.manifest_csv) or "REPLACE_ME" in str(cfg.labels_csv):
        print("labels_csv / manifest_csv are still REPLACE_ME: point both at the NEW behavior package folder.")
        return
    print("== labels.csv ==")
    labels = pd.read_csv(root / cfg.labels_csv)
    print(labels.dtypes.to_string(), "\n", labels.head(3).to_string(), "\n")
    print("== manifest ==")
    manifest = pd.read_csv(root / cfg.manifest_csv)
    print(manifest.dtypes.to_string(), "\n", manifest.head(3).to_string(), "\n")
    first = bd.read_jsonl(root / cfg.results_dir / cfg.source_splits[0] / "interactions.jsonl")[0]
    print("== interactions.jsonl: keys of first record ==\n", sorted(first), "\n")
    layer_dir = root / cfg.activations_dir / f"layer_{int(cfg.layers[0]):02d}"
    for split in cfg.source_splits:
        arr = bd.load_tensor(layer_dir / cfg.activation_file.format(split=split))
        print(f"== tensor {layer_dir.name}/{cfg.activation_file.format(split=split)}: {arr.shape} ==")
    print("\n== alignment check + eligible rows per partition and class ==")
    _, diag, _ = load_inputs(cfg, cols)
    print(json.dumps(diag, indent=2, default=str))
