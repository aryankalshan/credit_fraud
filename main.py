"""Run the experiment, or score transactions with its saved final model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf

from autoencoder import extract_features, train_autoencoder
from data import FEATURES, fit_preprocessor, load_data, split_data, transform
from experiments import EXPERIMENTS, evaluate, fuse, pca_features, train_classifier
from explain import explain_model, plot_pr


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def score_file(path, out):
    # Only load pickle artifacts created by this project and trusted by you.
    with (out / "preprocessor.pkl").open("rb") as handle:
        prep = pickle.load(handle)
    metadata = json.loads((out / "run.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(path)
    if set(frame.columns) not in [set(FEATURES), set(FEATURES + ["Class"])]:
        raise ValueError("Scoring CSV must have Time, V1...V28, Amount; Class is optional.")
    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    raw, scaled = transform(frame, prep)
    model = tf.keras.models.load_model(out / "autoencoder.keras", compile=False)
    encoder = tf.keras.Model(model.input, model.get_layer("latent").output)
    ae = extract_features(model, encoder, scaled, frame.index)
    features = pd.concat([raw, ae], axis=1)
    booster = lgb.Booster(model_file=str(out / "ae_full.txt"))
    if list(features.columns) != booster.feature_name():
        raise ValueError("Saved classifier feature order does not match extracted features.")
    scores = booster.predict(features)
    threshold = metadata["final_threshold"]
    result = pd.DataFrame({"row_id": frame.index, "fraud_score": scores,
                           "fraud_alert": (scores >= threshold).astype(int)})
    destination = out / "scored_transactions.csv"
    result.to_csv(destination, index=False)
    print(f"Scored {len(result):,} transactions -> {destination}")


def run(args):
    started = time.perf_counter()
    out = Path(args.out)
    if args.predict:
        score_file(args.predict, out)
        return
    if out.exists():
        raise FileExistsError(f"{out} already exists. Choose a new --out to preserve prior evidence.")
    frame, summary = load_data(args.data, args.download)
    parts = split_data(frame, args.split, args.seed)
    out.mkdir(parents=True)
    print("Dataset:", summary, flush=True)
    split_summary = {}
    for name, part in parts.items():
        split_summary[name] = {"rows": len(part), "frauds": int(part.Class.sum()),
                               "fraud_ratio": float(part.Class.mean()),
                               "time_min": float(part.Time.min()), "time_max": float(part.Time.max())}
    print("Splits:", split_summary, flush=True)
    pd.concat([pd.DataFrame({"split": name, "label": part.Class}, index=part.index)
               for name, part in parts.items()]).to_csv(out / "split_membership.csv")
    parts["train"][FEATURES].describe().T.to_csv(out / "training_feature_summary.csv")
    prep = fit_preprocessor(parts["train"])
    with (out / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(prep, handle)
    # The test features are not transformed or scored until all models/thresholds are frozen.
    raw, scaled, ae, principal = {}, {}, {}, {}
    labels = {name: part.Class.to_numpy() for name, part in parts.items()}
    for name in ["train", "validation"]:
        raw[name], scaled[name] = transform(parts[name], prep)
    print("Training the autoencoder on normal training rows only...", flush=True)
    ae_started = time.perf_counter()
    autoencoder, encoder, history = train_autoencoder(
        scaled["train"], labels["train"], scaled["validation"], labels["validation"],
        args.latent_dim, args.epochs, seed=args.seed)
    ae_seconds = time.perf_counter() - ae_started
    autoencoder.save(out / "autoencoder.keras")
    history.index = np.arange(1, len(history) + 1)
    history.index.name = "epoch"
    history.to_csv(out / "autoencoder_history.csv")
    pca = PCA(n_components=args.latent_dim, svd_solver="full")
    pca.fit(scaled["train"][labels["train"] == 0])
    with (out / "pca.pkl").open("wb") as handle:
        pickle.dump(pca, handle)
    for name in ["train", "validation"]:
        ae[name] = extract_features(autoencoder, encoder, scaled[name], raw[name].index)
        principal[name] = pca_features(pca, scaled[name], raw[name].index)
    models, validation = {}, {}
    for experiment in EXPERIMENTS:
        train = fuse(raw["train"], ae["train"], principal["train"], experiment)
        valid = fuse(raw["validation"], ae["validation"], principal["validation"], experiment)
        models[experiment], validation[experiment] = train_classifier(
            train, labels["train"], valid, labels["validation"], args.min_precision, args.seed)
        models[experiment].booster_.save_model(str(out / f"{experiment}.txt"))
        print(f"Validation {experiment}: {validation[experiment]}", flush=True)
    pd.DataFrame(validation).T.rename_axis("model").to_csv(out / "validation_results.csv")
    if args.validation_only:
        print(f"Validation-only run complete in {out}; test features were not scored.")
        return
    print("All models and thresholds frozen. Evaluating the held-out test set...", flush=True)
    raw["test"], scaled["test"] = transform(parts["test"], prep)
    ae["test"] = extract_features(autoencoder, encoder, scaled["test"], raw["test"].index)
    principal["test"] = pca_features(pca, scaled["test"], raw["test"].index)
    results, predictions, default_results = {}, {}, {}
    for experiment, model in models.items():
        features = fuse(raw["test"], ae["test"], principal["test"], experiment)
        predictions[experiment] = model.predict_proba(features)[:, 1]
        results[experiment] = evaluate(labels["test"], predictions[experiment],
                                       validation[experiment]["threshold"])
        default_results[experiment] = evaluate(labels["test"], predictions[experiment], 0.5)
    table = pd.DataFrame(results).T.rename_axis("model")
    table.to_csv(out / "results.csv")
    pd.DataFrame(default_results).T.rename_axis("model").to_csv(out / "default_threshold_results.csv")
    pd.DataFrame({"label": labels["test"], **predictions}, index=parts["test"].index).to_csv(
        out / "test_predictions.csv")
    error_rows = []
    for split in ["validation", "test"]:
        for label in [0, 1]:
            error = ae[split].loc[labels[split] == label, "ae_error"]
            error_rows.append({"split": split, "class": label, "count": len(error),
                               "mean": float(error.mean()), "median": float(error.median()),
                               "p90": float(error.quantile(0.9)), "p99": float(error.quantile(0.99))})
    pd.DataFrame(error_rows).to_csv(out / "reconstruction_summary.csv", index=False)
    plot_pr(labels["test"], predictions, out)
    final_features = fuse(raw["test"], ae["test"], principal["test"], "ae_full")
    print("Explaining the prespecified final fused model with SHAP...", flush=True)
    top_features, local = explain_model(models["ae_full"], final_features, labels["test"],
                                        predictions["ae_full"], results["ae_full"]["threshold"], out,
                                        args.seed)
    final_ap = results["ae_full"]["pr_auc_ap"]
    comparison = {"ae_minus_pca_ap": final_ap - results["pca"]["pr_auc_ap"],
                  "ae_minus_pca_full_ap": final_ap - results["pca_full"]["pr_auc_ap"],
                  "relative_improvement_percent": 100 * (final_ap / results["pca"]["pr_auc_ap"] - 1)}
    versions = {name: importlib.metadata.version(name) for name in
                ["tensorflow", "lightgbm", "numpy", "pandas", "scikit-learn", "shap", "matplotlib"]}
    with Path(args.data).open("rb") as handle:
        dataset_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {"config": vars(args), "dataset": summary, "dataset_sha256": dataset_hash,
                "splits": split_summary, "versions": versions,
                "autoencoder": {"architecture": [30, 32, args.latent_dim, 32, 30],
                                "normal_training_rows": int(np.sum(labels["train"] == 0)),
                                "fraud_training_rows": 0, "epochs_run": len(history),
                                "best_epoch": int(history.val_loss.idxmin()),
                                "first_validation_loss": float(history.val_loss.iloc[0]),
                                "best_validation_loss": float(history.val_loss.min()),
                                "seconds": ae_seconds},
                "comparison": comparison, "final_threshold": results["ae_full"]["threshold"],
                "final_feature_names": list(final_features.columns), "shap_top_features": top_features,
                "total_seconds": time.perf_counter() - started}
    write_json(out / "run.json", metadata)
    lines = ["# Measured experiment results", "", f"Split: {args.split}; seed: {args.seed}. "
             f"Validation precision floor: {args.min_precision:.0%}. PR-AUC is average precision (AP).", "",
             "| Model / features | Recall | Precision | PR-AUC (AP) | Threshold |",
             "|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append(f"| {name} | {row['recall']:.4f} | {row['precision']:.4f} | "
                     f"{row['pr_auc_ap']:.4f} | {row['threshold']:.6g} |")
    lines += ["", f"AE minus PCA AP: {comparison['ae_minus_pca_ap']:+.6f}; "
              f"relative change: {comparison['relative_improvement_percent']:+.2f}%.",
              f"AE minus PCA with reconstruction error AP: {comparison['ae_minus_pca_full_ap']:+.6f}.", "",
              "These are single-split point estimates, not evidence of universal or statistically significant superiority.",
              "Test precision may fall below the validation floor. Scores are not calibrated fraud probabilities."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table[["recall", "precision", "pr_auc_ap", "threshold"]].to_string(), flush=True)
    print(f"Finished in {metadata['total_seconds']:.1f}s. Evidence saved to {out}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--split", choices=["temporal", "stratified"], default="temporal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--min-precision", type=float, default=0.8)
    parser.add_argument("--predict", metavar="CSV", help="Score a CSV using the saved models in --out")
    parser.add_argument("--validation-only", action="store_true",
                        help="Stop before test scoring; use for development experiments")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or not 0 < arguments.min_precision <= 1 or not 0 < arguments.latent_dim < 30:
        parser.error("Require epochs >= 1, 0 < min-precision <= 1, and 0 < latent-dim < 30.")
    run(arguments)
