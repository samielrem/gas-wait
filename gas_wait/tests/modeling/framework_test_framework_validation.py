"""Validate framework output against saved exp01 weekly dataset."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
sys.path.insert(0, str(SRC))

from modeling.backtest import BacktestConfig, run_simple_backtest
from modeling.config import FeatureGroup, resolve_feature_columns
from modeling.signals import generate_signal
from modeling.weekly_pipeline import build_weekly_eia_dataset

COMPARE_COLS = resolve_feature_columns(list(FeatureGroup))


class FrameworkValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        saved_path = PROCESSED / "weekly_model_dataset.csv"
        if not saved_path.exists():
            raise unittest.SkipTest("weekly_model_dataset.csv not found")
        cls.saved = pd.read_csv(
            saved_path,
            parse_dates=[
                "prediction_date",
                "prediction_ts_utc",
                "target_timestamp",
                "retail_monday",
                "retail_feature_timestamp",
                "spot_feature_timestamp",
                "inventory_feature_timestamp",
                "inventory_release_ts_utc",
            ],
        )
        cls.built = build_weekly_eia_dataset(PROCESSED)

    def test_same_row_count(self) -> None:
        self.assertEqual(len(self.built), len(self.saved))

    def test_feature_values_match_exp01(self) -> None:
        merged = self.saved.merge(
            self.built,
            on="prediction_date",
            suffixes=("_saved", "_built"),
        )
        self.assertEqual(len(merged), len(self.saved))
        rtol = 1e-9
        atol = 1e-9
        for col in COMPARE_COLS:
            a = merged[f"{col}_saved"].to_numpy(dtype=float)
            b = merged[f"{col}_built"].to_numpy(dtype=float)
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=col)

    def test_signals_on_momentum_baseline(self) -> None:
        row = self.saved.iloc[100]
        sig = generate_signal(float(row["retail_d7"]))
        self.assertIn(sig.signal, ("WAIT", "FILL UP", "NO CLEAR SIGNAL"))

    def test_backtest_demo_no_training(self) -> None:
        result = run_simple_backtest(
            self.saved,
            lambda d: d["retail_d7"],
            BacktestConfig(train_fraction=0.7),
        )
        self.assertTrue(result.is_chronological)
        self.assertGreater(result.predictions["signal"].ne("NO CLEAR SIGNAL").sum(), 0)


if __name__ == "__main__":
    unittest.main()
