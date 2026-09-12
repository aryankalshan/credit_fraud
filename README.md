# Credit Card Fraud Detection

A compact offline experiment combining a **TensorFlow autoencoder** with **LightGBM**. The neural network learns normal transaction behavior; the tree classifier uses that representation to detect fraud. Five implementation files and one small correctness-check file.

```text
30 transaction features ──────────────────────────────┐
       ↓ training-only imputation + standardization   │
Normal-only autoencoder: 30 → 32 → 8 → 32 → 30       │
       ↓ 8 latent codes + 1 reconstruction error       │
       └────────── raw-feature fusion ←───────────────┘
                           ↓
                    LightGBM: 39 inputs
                           ↓
             fraud score → validation threshold → alert
```

## Run

Use **Python 3.12**. A CPU is sufficient; the project has no GPU requirement.
The local `.venv` has already been prepared for this workspace; skip installation to run it directly.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use these commands from the project folder (no environment activation required):

```powershell
.\.venv\Scripts\python.exe -m unittest checks -v
.\.venv\Scripts\python.exe main.py --download --out outputs_reproduce
```

The first run downloads the dataset if missing. Alternatively, put the original `creditcard.csv` in `data/` or pass `--data PATH`. Expected output: dataset/split counts, per-epoch losses, validation metrics for six models, and a held-out results table. The command above writes a fresh `outputs_reproduce/`; the completed experiment is in `outputs_final/`.

For a fresh experiment, use a new output folder:

```powershell
.\.venv\Scripts\python.exe main.py --out outputs_second
```

Scoring with the saved final model requires the same 30 feature columns; their CSV order may differ. `Class` is optional and ignored during scoring.

```powershell
.\.venv\Scripts\python.exe main.py --predict data/creditcard.csv --out outputs_final
```

This writes `outputs_final/scored_transactions.csv`. Scoring the training dataset is **not** a new evaluation. Only load saved pickle files you trust.

## Dataset and splits

The [ULB/Worldline credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), available through [TensorFlow's documented mirror](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data), contains 284,807 transactions and 492 fraud labels before deduplication. `Class=1` means fraud. Inputs are `Time`, `Amount`, and anonymized `V1`–`V28`; the V features were already PCA-transformed by the dataset provider. This project's PCA is an additional baseline over the supplied features; the original financial variables are unavailable.

Exact duplicate records are removed before splitting. Default: approximately **60% training / 20% validation / 20% test in time order**, keeping equal timestamps together. The optional `--split stratified` uses the same proportions with seed 42, but does not test future-period generalization. Do not choose between splits based on the most flattering test result.

Median imputation and standardization are fitted on training rows only. LightGBM receives the original feature values after imputation; the autoencoder and PCA receive standardized inputs. No resampling is applied. CSV row IDs are retained to check feature alignment. The supplied upstream PCA cannot be audited for leakage; there are no customer IDs for customer-disjoint testing.

## Models and threshold

- **Autoencoder:** 30 → 32 ReLU → 8 linear → 32 ReLU → 30 linear. Adam at 0.001, MSE loss, batch size 512, up to 30 epochs, normal-validation early stopping with patience 5 and best weights restored. Only label-0 training rows enter gradient updates. Linear output permits negative standardized values. Error is mean squared error across all 30 dimensions.
- **PCA:** 8 components, fitted on the same normal training rows. Equal dimensionality and preprocessing make comparison more controlled.
- **LightGBM:** up to 500 trees, learning rate 0.04, 15 leaves, minimum 40 rows per leaf, L2 regularization 2, and `max_delta_step=5` to cap per-tree log-odds updates at 0.2. Positive weight is the square root of the training normal/fraud ratio. This moderate weight and all settings are identical across experiments. Early stopping uses validation average precision with patience 50; selected tree counts may differ.
- **Threshold:** among all distinct validation scores, maximize recall subject to precision ≥ 80%; break ties by higher precision then higher threshold. If infeasible, flag that explicitly and use an all-negative validation threshold. The precision floor is an illustrative operating choice, not a business-approved cost model or a test-set guarantee. Weighted classifier scores are not calibrated probabilities.

Six experiments: raw; raw + PCA; raw + PCA + PCA error; raw + AE error; raw + AE latent; raw + AE latent + AE error. The extra PCA-error control matches the final model's 39 inputs. This is a small ablation, not a hyperparameter search. `ae_full` is the prespecified final architecture, whether or not it wins.

For development, append `--validation-only --out outputs_development` to stop before any test scoring. The initial uncapped run selected only two trees for several variants on validation; the update cap was chosen from this diagnostic before inspecting test results and then applied uniformly. The initial artifacts remain in `outputs/`; the corrected final experiment is in `outputs_final/`.

## Evaluation and measured results

Measured on **56,746 chronological test transactions, including 74 frauds**, after training on 170,235 rows and validating on 56,745. Deduplication left 283,726 rows, 473 frauds (0.1667%), and no missing values. All alert thresholds were selected on validation data.

| LightGBM inputs | Recall | Precision | PR-AUC (AP) |
|---|---:|---:|---:|
| Raw | 0.7432 | 0.8462 | 0.7935 |
| Raw + PCA | 0.7027 | 0.8814 | 0.7881 |
| Raw + PCA + PCA error | 0.7027 | 0.8667 | 0.7829 |
| Raw + AE error | 0.7297 | 0.8710 | 0.7917 |
| Raw + AE latent | 0.7432 | 0.8871 | **0.8003** |
| Raw + AE latent + AE error (final architecture) | 0.7027 | 0.8966 | 0.7842 |

**Full fusion did not outperform the standard raw + PCA baseline:** AP difference −0.003860 (−0.49% relative). It exceeded the matched PCA-plus-error control by only +0.001351. The latent-only ablation was strongest; adding reconstruction error did not improve its AP. These findings do not support an unqualified CV claim that the complete pipeline outperformed PCA.

Final threshold **0.705815** caught 52 frauds, missed 22, and generated 6 false positives. Threshold 0.5 caught 56 frauds with 13 false positives; the validation precision constraint led to a more conservative operating point. The AE trained on **169,893 normal rows and zero fraud rows** for 30 epochs; normal-validation MSE fell from 0.8505 to 0.3808. Median test reconstruction error was 0.2257 for normal transactions and 4.0229 for fraud, with overlap. AE training took 58.1 seconds; the full corrected run took 129.4 seconds on this machine, excluding installation/download.

See `outputs_final/RESULTS.md`, `outputs_final/precision_recall.png`. Six correctness checks passed. Reloading the saved pipeline reproduced 100 held-out scores with reversed input-column order at a tolerance of 1e-12; global/local SHAP additivity also passed.

PR-AUC is explicitly reported as **average precision (AP)** using scikit-learn, not trapezoidal PR area. Recall, precision, F1, ROC-AUC, and confusion counts are also saved. Test data are scored only after every classifier and validation threshold is frozen. The default-0.5 comparison makes the alert tradeoff visible; thresholds do not change AP.

## SHAP and saved evidence

SHAP TreeExplainer explains the final `ae_full` LightGBM model in raw log-odds space. Global importance uses mean absolute SHAP over up to 500 randomly sampled test rows at natural prevalence. Local examples show the highest-scoring correctly detected fraud and false positive, if present. Additivity is checked against the classifier's raw score. Positive SHAP values raise the fraud score relative to the baseline; they do not establish causality. Latent dimensions and anonymized V features have no defensible financial names.

Measured global leaders: V14, V4, Time, V8, and reconstruction error. In the detected fraud at row 229712, reconstruction error contributed +2.0682 log odds; in the legitimate alert at row 274771 it contributed +2.3185. This illustrates both useful anomaly information and false-positive risk. See `outputs_final/shap_global.png`, `outputs_final/shap_true_positive.png`, and `outputs_final/shap_false_positive.png`.

`outputs_final/` contains the PR curve, global/local SHAP plots and attribution values, reconstruction-error summaries by class, training loss history, validation/test metrics, default-threshold metrics, test scores, split membership, and saved models. `run.json` records configuration, package versions, dataset SHA-256, training duration, and AE/PCA AP differences. `RESULTS.md` is the generated table.

## Files and limitations

| File | Purpose |
|---|---|
| `data.py` | Loading, deduplication, splits, preprocessing |
| `autoencoder.py` | Normal-only training and AE features |
| `experiments.py` | Feature fusion, PCA features, LightGBM, metrics, threshold |
| `explain.py` | PR curve and global/local SHAP |
| `main.py` | Experiment runner and saved-model scoring |
| `checks.py` | Leakage, alignment, threshold, and metric checks |

Limitations: a short historical anonymized dataset, few fraud labels, concept drift, false positives, no operational cost model, and one chronological holdout. An autoencoder can reconstruct fraud well or flag unusual legitimate transactions. Better reconstruction need not yield better classification. A single test-set difference does not establish significance or general superiority. This is an offline student ML project.

Library references: [TensorFlow autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder), [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

## Repository contents

This repository includes source code, tests, dependency manifests, setup documentation, and required example data. Downloaded datasets, trained models, caches, generated reports, benchmark logs, and personal interview preparation are excluded. Result paths mentioned above are generated locally by the documented commands; previously reported measurements describe the original local runs.
