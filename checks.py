"""Small correctness checks. Run: python -m unittest checks -v."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data import FEATURES, fit_preprocessor, split_data, transform
from experiments import evaluate, fuse, select_threshold


class PipelineChecks(unittest.TestCase):
    def test_autoencoder_excludes_fraud_and_extracts_correct_mse(self):
        import tensorflow as tf
        from autoencoder import extract_features, train_autoencoder

        rng = np.random.default_rng(5)
        train = rng.normal(size=(24, 4)).astype("float32")
        labels = np.array([0] * 20 + [1] * 4)
        seen = {}
        original_fit = tf.keras.Model.fit

        def capture(model, x, y, **kwargs):
            seen["train"] = x.copy()
            seen["valid"] = kwargs["validation_data"][0].copy()
            return original_fit(model, x, y, **kwargs)

        with patch.object(tf.keras.Model, "fit", new=capture):
            model, encoder, _ = train_autoencoder(train, labels, train, labels,
                                                   latent_dim=2, epochs=1)
        np.testing.assert_array_equal(seen["train"], train[:20])
        np.testing.assert_array_equal(seen["valid"], train[:20])
        features = extract_features(model, encoder, train, pd.Index(range(24)))
        expected = np.mean((train - model(train, training=False).numpy()) ** 2, axis=1)
        np.testing.assert_allclose(features.ae_error, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(features.shape, (24, 3))

    def test_threshold_matches_exhaustive_search_with_ties(self):
        labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.2, 0.5, 0.6, 0.6, 0.9, 0.95])
        threshold, feasible = select_threshold(labels, scores, 0.5)
        candidates = [evaluate(labels, scores, t) for t in np.unique(scores)]
        candidates = [m for m in candidates if m["precision"] >= 0.5]
        best = max(candidates, key=lambda m: (m["recall"], m["precision"], m["threshold"]))
        self.assertTrue(feasible)
        self.assertEqual(threshold, best["threshold"])

    def test_infeasible_precision_floor_is_explicit(self):
        threshold, feasible = select_threshold([1, 0], np.array([0.1, 0.9]), 0.8)
        self.assertFalse(feasible)
        self.assertGreater(threshold, 0.9)

    def test_metric_conventions(self):
        metrics = evaluate([1, 0, 1, 0], np.array([0.9, 0.8, 0.7, 0.1]), 0.75)
        self.assertAlmostEqual(metrics["pr_auc_ap"], (1 + 2 / 3) / 2)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["recall"], 0.5)

    def test_fusion_rejects_row_misalignment(self):
        raw = pd.DataFrame({"x": [1, 2]}, index=[3, 7])
        ae = pd.DataFrame({"ae_1": [4, 5], "ae_error": [0.1, 0.2]}, index=[7, 3])
        pca = pd.DataFrame({"pca_1": [6, 7], "pca_error": [0.2, 0.3]}, index=[3, 7])
        with self.assertRaises(ValueError):
            fuse(raw, ae, pca, "ae_full")

    def test_split_and_training_only_scaling(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(rng.normal(size=(600, 30)), columns=FEATURES)
        frame["Time"] = np.arange(600) // 2
        frame["Class"] = np.tile([0, 0, 1], 200)
        parts = split_data(frame)
        self.assertLess(parts["train"].Time.max(), parts["validation"].Time.min())
        self.assertLess(parts["validation"].Time.max(), parts["test"].Time.min())
        prep = fit_preprocessor(parts["train"])
        before = prep["scaler"].mean_.copy()
        held_out = parts["test"].copy()
        held_out["V1"] = 1e6
        held_out.loc[held_out.index[0], "Amount"] = np.nan
        raw, scaled = transform(held_out, prep)
        np.testing.assert_array_equal(before, prep["scaler"].mean_)
        self.assertEqual(list(raw.index), list(held_out.index))
        self.assertTrue(np.isfinite(scaled).all())
        self.assertGreater(scaled[:, 1].min(), 100)


if __name__ == "__main__":
    unittest.main()
