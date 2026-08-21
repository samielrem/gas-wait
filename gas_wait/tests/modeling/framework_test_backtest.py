"""Backtest engine tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from modeling.backtest import BacktestConfig, chronological_split, run_simple_backtest


class BacktestTests(unittest.TestCase):
    def test_chronological_split_order(self) -> None:
        df = pd.DataFrame(
            {
                "prediction_date": pd.date_range("2020-01-07", periods=10, freq="7D"),
                "target": range(10),
                "retail_d7": range(10),
            }
        )
        train, test, _ = chronological_split(df, "prediction_date", 0.7)
        self.assertTrue(train["prediction_date"].max() < test["prediction_date"].min())

    def test_simple_backtest_momentum_is_chronological(self) -> None:
        df = pd.DataFrame(
            {
                "prediction_date": pd.date_range("2020-01-07", periods=20, freq="7D"),
                "target": [0.01 * ((-1) ** i) for i in range(20)],
                "retail_d7": [0.02 * ((-1) ** i) for i in range(20)],
            }
        )
        result = run_simple_backtest(
            df,
            lambda d: d["retail_d7"],
            BacktestConfig(train_fraction=0.5),
        )
        self.assertTrue(result.is_chronological)
        self.assertIn("signal", result.predictions.columns)


if __name__ == "__main__":
    unittest.main()
