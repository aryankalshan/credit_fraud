"""Stage 2: controlled LightGBM comparisons and validation-only thresholds."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

EXPERIMENTS = ["raw", "pca", "pca_full", "ae_error", "ae_latent", "ae_full"]


def fuse(raw, ae, pca, experiment):
    if not raw.index.equals(ae.index) or not raw.index.equals(pca.index):
        raise ValueError("Transaction rows are misaligned.")
    if experiment == "raw":
        return raw
    extra = {"pca": pca.drop(columns="pca_error"), "pca_full": pca,
             "ae_error": ae[["ae_error"]], "ae_latent": ae.drop(columns="ae_error"),
             "ae_full": ae}[experiment]
    return pd.concat([raw, extra], axis=1)


def pca_features(pca, scaled, index):
    codes = pca.transform(scaled)
    features = pd.DataFrame(codes, index=index,
                            columns=[f"pca_{i + 1}" for i in range(codes.shape[1])])
    features["pca_error"] = np.mean((scaled - pca.inverse_transform(codes)) ** 2, axis=1)
    return features


def select_threshold(labels, scores, min_precision=0.8):
    """Maximize recall subject to an empirical validation precision floor."""
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    feasible = np.flatnonzero(precision[:-1] >= min_precision)
    if len(feasible) == 0:
        # Do not pretend an all-negative prediction meets a precision constraint.
        return float(np.nextafter(np.max(scores), np.inf)), False
    best = max(feasible, key=lambda i: (recall[i], precision[i], thresholds[i]))
    return float(thresholds[best]), True


def evaluate(labels, scores, threshold):
    alerts = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, alerts, labels=[0, 1]).ravel()
    return {"recall": float(recall_score(labels, alerts, zero_division=0)),
            "precision": float(precision_score(labels, alerts, zero_division=0)),
            "pr_auc_ap": float(average_precision_score(labels, scores)),
            "f1": float(f1_score(labels, alerts, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)), "threshold": threshold,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def train_classifier(train, labels, valid, valid_labels, min_precision=0.8, seed=42):
    # Moderate weighting avoids making the very small fraud class dominate each split.
    weight = float(np.sqrt(np.sum(labels == 0) / np.sum(labels == 1)))
    model = lgb.LGBMClassifier(
        objective="binary", metric="None", n_estimators=500, learning_rate=0.04,
        num_leaves=15, max_depth=-1, min_child_samples=40, reg_lambda=2.0,
        max_delta_step=5.0,  # Cap each tree's log-odds update under rare-class weighting.
        scale_pos_weight=weight, random_state=seed, n_jobs=4,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(train, labels, eval_set=[(valid, valid_labels)],
              eval_metric=lambda y, p: ("average_precision", average_precision_score(y, p), True),
              callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
    scores = model.predict_proba(valid)[:, 1]
    threshold, feasible = select_threshold(valid_labels, scores, min_precision)
    metrics = evaluate(valid_labels, scores, threshold)
    metrics.update(precision_floor_met=feasible, best_iteration=model.best_iteration_,
                   scale_pos_weight=weight)
    return model, metrics
