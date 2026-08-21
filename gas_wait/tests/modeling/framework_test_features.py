"""Feature calculation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from modeling.features import (
    add_crude_features,
    add_retail_momentum_weekly,
    calendar_day_change,
    session_changes,
)


class FeatureTests(unittest.TestCase):
    def test_session_changes_use_past_rows_only(self) -> None:
        df = pd.DataFrame(
            {
                "observation_date": pd.date_range("2024-01-02", periods=5, freq="B"),
                "wti": [70.0, 71.0, 70.5, 72.0, 71.0],
            }
        )
        out = session_changes(df, "wti", (1, 3), "wti")
        self.assertTrue(np.isnan(out.loc[0, "wti_d1"]))
        self.assertAlmostEqual(out.loc[1, "wti_d1"], 1.0)
        self.assertAlmostEqual(out.loc[3, "wti_d3"], 2.0)

    def test_calendar_day_change_no_interpolation(self) -> None:
        df = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"]),
                "wti": [70.0, 71.0, 72.0],
            }
        )
        out = calendar_day_change(df, "wti", 7, "wti")
        # Last row uses price on or before 2024-01-01 -> NaN (no prior row)
        self.assertTrue(np.isnan(out.loc[2, "wti_chg_7cal"]))

    def test_retail_weekly_momentum_gap_check(self) -> None:
        df = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-22"]),
                "retail_price": [3.0, 3.1, 3.2],
            }
        )
        out = add_retail_momentum_weekly(df, "retail_price")
        self.assertAlmostEqual(out.loc[1, "retail_d7"], 0.1)
        self.assertTrue(np.isnan(out.loc[2, "retail_d14"]))

    def test_crude_volatility_requires_window(self) -> None:
        dates = pd.date_range("2024-01-02", periods=25, freq="B")
        prices = np.linspace(70, 72, 25)
        df = pd.DataFrame({"observation_date": dates, "wti": prices})
        out = add_crude_features(df, include_extended=False)
        self.assertFalse(np.isnan(out["wti_vol_20"].iloc[-1]))
        self.assertTrue(out["wti_vol_20"].iloc[:15].isna().all())


if __name__ == "__main__":
    unittest.main()
