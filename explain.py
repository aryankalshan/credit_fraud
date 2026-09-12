"""A small PR plot and SHAP explanations of the final fused classifier."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_pr(labels, predictions, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ["raw", "pca", "ae_full"]:
        precision, recall, _ = precision_recall_curve(labels, predictions[name])
        ap = average_precision_score(labels, predictions[name])
        ax.step(recall, precision, where="post", label=f"{name} (AP {ap:.3f})")
    ax.axhline(np.mean(labels), color="gray", linestyle=":", label="Fraud prevalence")
    ax.set(xlabel="Fraud recall", ylabel="Fraud precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Held-out test precision–recall curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", dpi=160)
    plt.close(fig)


def explain_model(model, features, labels, scores, threshold, out, seed=42):
    # Random sample retains natural prevalence; local alerts are explained separately.
    sample = features.sample(n=min(500, len(features)), random_state=seed)
    explainer = shap.TreeExplainer(model, model_output="raw",
                                   feature_perturbation="tree_path_dependent")
    explanation = explainer(sample)
    raw_scores = model.predict(sample, raw_score=True)
    np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1),
                               raw_scores, rtol=1e-5, atol=1e-5)
    importance = pd.Series(np.abs(explanation.values).mean(axis=0),
                           index=features.columns).sort_values(ascending=False)
    importance.rename_axis("feature").rename("mean_abs_shap").to_csv(out / "shap_global.csv")
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Global SHAP importance · 500 test rows at most")
    plt.tight_layout()
    plt.savefig(out / "shap_global.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    alerts = scores >= threshold
    cases = {"true_positive": np.flatnonzero((labels == 1) & alerts),
             "false_positive": np.flatnonzero((labels == 0) & alerts)}
    details = []
    for name, candidates in cases.items():
        if len(candidates) == 0:
            details.append({"case": name, "status": "No such alert on the test set."})
            continue
        position = candidates[np.argmax(scores[candidates])]
        row = features.iloc[[position]]
        local = explainer(row)
        np.testing.assert_allclose(local.base_values + local.values.sum(axis=1),
                                   model.predict(row, raw_score=True), atol=1e-5)
        shap.plots.waterfall(local[0], max_display=12, show=False)
        plt.title(f"{name.replace('_', ' ').title()} · row {row.index[0]}")
        plt.savefig(out / f"shap_{name}.png", dpi=160, bbox_inches="tight")
        plt.close("all")
        values = pd.DataFrame({"feature": features.columns, "value": row.iloc[0].to_numpy(),
                               "shap_log_odds": local.values[0]})
        values = values.iloc[np.argsort(-np.abs(values.shap_log_odds.to_numpy()))]
        values.to_csv(out / f"shap_{name}.csv", index=False)
        details.append({"case": name, "row_id": int(row.index[0]),
                        "label": int(labels[position]), "score": float(scores[position]),
                        "threshold": threshold, "base_log_odds": float(local.base_values[0]),
                        "top_contributions": values.head(5).to_dict("records")})
    (out / "shap_local.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return importance.head(5).to_dict(), details
