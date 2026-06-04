import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.feature_engineering import FeatureEngineer
from data.feature_schema import FeatureSchema, FeatureSchemaError, FeatureValidator


def make_raw_bars(rows=700):
    rng = np.random.default_rng(42)
    base = 2000 + np.cumsum(rng.normal(0, 0.8, rows))
    open_ = base + rng.normal(0, 0.2, rows)
    close = base + rng.normal(0, 0.2, rows)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.8, rows)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.8, rows)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 1000, rows),
            "spread": rng.integers(10, 50, rows),
        }
    )


class FeatureEngineeringSafetyTests(unittest.TestCase):
    def test_feature_count_is_stable_across_repeated_calls(self):
        raw = make_raw_bars()
        original_columns = list(raw.columns)
        engineer = FeatureEngineer(include_higher_timeframe=False)
        counts = []
        model_counts = []

        for _ in range(5):
            features = engineer.add_all_features(raw)
            counts.append(len(engineer.get_feature_columns()))
            model_counts.append(len(engineer.get_model_input_columns(features)))

        self.assertEqual(counts, [71, 71, 71, 71, 71])
        self.assertEqual(model_counts, [76, 76, 76, 76, 76])
        self.assertEqual(list(raw.columns), original_columns)

    def test_duplicate_input_columns_raise(self):
        raw = make_raw_bars()
        duplicate = pd.concat([raw, raw[["open"]]], axis=1)
        with self.assertRaisesRegex(ValueError, "duplicate columns"):
            FeatureEngineer(include_higher_timeframe=False).add_all_features(duplicate)

    def test_missing_raw_columns_raise(self):
        raw = make_raw_bars().drop(columns=["close"])
        with self.assertRaisesRegex(ValueError, "missing required raw columns"):
            FeatureEngineer(include_higher_timeframe=False).add_all_features(raw)

    def test_recursive_feature_input_raises(self):
        raw = make_raw_bars()
        features = FeatureEngineer(include_higher_timeframe=False).add_all_features(raw)
        with self.assertRaisesRegex(ValueError, "raw OHLCV data only"):
            FeatureEngineer(include_higher_timeframe=False).add_all_features(features)

    def test_higher_timeframe_features_are_stable_and_numeric(self):
        raw = make_raw_bars(rows=3000)
        engineer = FeatureEngineer(include_higher_timeframe=True)
        features = engineer.add_all_features(raw)
        htf_cols = [c for c in features.columns if c.startswith("htf_")]

        self.assertEqual(len(htf_cols), 20)
        self.assertIn("htf_h1_ema_200", htf_cols)
        self.assertIn("htf_m15_breakout_position", htf_cols)
        self.assertIn("htf_h4_atr_regime_ratio", htf_cols)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(features[c]) for c in htf_cols))
        self.assertFalse(features[htf_cols].isna().any().any())

        repeat = engineer.add_all_features(raw)
        self.assertEqual(len(engineer.get_feature_columns()), 91)
        self.assertEqual(len(engineer.get_model_input_columns(repeat)), 96)

    def test_higher_timeframe_alignment_uses_closed_candles_only(self):
        raw = make_raw_bars(rows=3000)
        engineer = FeatureEngineer(include_higher_timeframe=True)
        h1 = engineer._build_h1_ema_trend(raw)
        row_before_h1_close = raw.loc[raw["time"] == pd.Timestamp("2026-01-01 00:55:00", tz="UTC")]
        row_at_h1_close = raw.loc[raw["time"] == pd.Timestamp("2026-01-01 01:00:00", tz="UTC")]

        merged_before = pd.merge_asof(
            row_before_h1_close.sort_values("time"),
            h1.sort_values("time"),
            on="time",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_at_close = pd.merge_asof(
            row_at_h1_close.sort_values("time"),
            h1.sort_values("time"),
            on="time",
            direction="backward",
            allow_exact_matches=True,
        )

        self.assertTrue(pd.isna(merged_before["htf_h1_ema_20"].iloc[0]))
        expected = h1.loc[h1["time"] == pd.Timestamp("2026-01-01 01:00:00", tz="UTC"), "htf_h1_ema_20"].iloc[0]
        self.assertEqual(merged_at_close["htf_h1_ema_20"].iloc[0], expected)


class FeatureSchemaValidationTests(unittest.TestCase):
    def setUp(self):
        raw = make_raw_bars()
        self.features = FeatureEngineer(include_higher_timeframe=False).add_all_features(raw)
        self.feature_names = [
            c for c in self.features.columns
            if c not in {"time", "label", "label_name", "close"}
        ]
        self.schema = FeatureSchema.from_feature_names(self.feature_names)

    def test_valid_schema_aligns_in_training_order(self):
        X = FeatureValidator.align_matrix(self.features, self.schema)
        self.assertEqual(list(X.columns), self.feature_names)
        self.assertEqual(X.shape[1], len(self.feature_names))

    def test_missing_feature_raises(self):
        bad = self.features.drop(columns=[self.feature_names[-1]])
        with self.assertRaisesRegex(FeatureSchemaError, "Missing model features"):
            FeatureValidator.validate_dataframe(bad, self.schema)

    def test_extra_feature_raises(self):
        bad = self.features.copy()
        bad["unexpected_feature"] = 1.0
        with self.assertRaisesRegex(FeatureSchemaError, "Extra feature columns"):
            FeatureValidator.validate_dataframe(bad, self.schema)

    def test_reordered_feature_raises(self):
        cols = list(self.features.columns)
        i = cols.index(self.feature_names[0])
        j = cols.index(self.feature_names[1])
        cols[i], cols[j] = cols[j], cols[i]
        bad = self.features[cols]
        with self.assertRaisesRegex(FeatureSchemaError, "column order"):
            FeatureValidator.validate_dataframe(bad, self.schema)

    def test_nan_and_inf_raise(self):
        nan_bad = self.features.copy()
        nan_bad.loc[nan_bad.index[-1], self.feature_names[0]] = np.nan
        with self.assertRaisesRegex(FeatureSchemaError, "NaN values"):
            FeatureValidator.validate_dataframe(nan_bad, self.schema)

        inf_bad = self.features.copy()
        inf_bad.loc[inf_bad.index[-1], self.feature_names[0]] = np.inf
        with self.assertRaisesRegex(FeatureSchemaError, "Infinite values"):
            FeatureValidator.validate_dataframe(inf_bad, self.schema)


class LiveInferenceIntegrationTests(unittest.TestCase):
    def test_saved_model_schema_predicts_on_cached_m5_features(self):
        model_path = Path("saved_models/USTEC_X100/M5/candidate_xgboost_model")
        csv_path = Path("data/USTEC_X100/M5/ustec_x100_m5_12m.csv")
        if not model_path.exists() or not csv_path.exists():
            self.skipTest("Saved USTEC_X100 M5 model/data artifact is not available")

        try:
            from scripts.backtest_run import load_saved_model

            model = load_saved_model(str(model_path))
        except Exception as exc:
            self.skipTest(f"Saved model could not be loaded in this environment: {exc}")

        raw = pd.read_csv(csv_path).tail(700).reset_index(drop=True)
        raw["time"] = pd.to_datetime(raw["time"], utc=True)
        
        # Dynamically determine if the saved model requires higher timeframe features
        include_htf = any(str(name).startswith("htf_") for name in model._feature_names)
        features = FeatureEngineer(include_higher_timeframe=include_htf).add_all_features(raw)
        schema = getattr(model, "_feature_schema", None) or FeatureSchema.from_feature_names(model._feature_names)

        X = FeatureValidator.align_matrix(features, schema)
        self.assertEqual(X.shape[1], len(model._feature_names))

        proba = model.predict_proba(X.tail(1))
        self.assertIn(proba.shape[1], {2, 3})
        self.assertTrue(np.isfinite(proba).all())


if __name__ == "__main__":
    unittest.main()
