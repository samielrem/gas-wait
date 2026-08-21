"""Leakage and point-in-time tests for the first modeling experiment. No live API."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modeling.build_features import (
    ALL_MODEL_FEATURES,
    _assert_no_leakage,
    _wpsr_release_ts,
    build_weekly_model_dataset,
)


class ModelingLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = build_weekly_model_dataset()
        cls.df = result.frame
        cls.dropped = result.dropped

    def test_dataset_is_nonempty(self) -> None:
        self.assertGreater(len(self.df), 200)
        for col in ALL_MODEL_FEATURES:
            self.assertIn(col, self.df.columns)
            self.assertFalse(self.df[col].isna().any())

    def test_no_la_rbob_column(self) -> None:
        joined = " ".join(self.df.columns)
        self.assertNotIn("la_rbob", joined)
        self.assertNotIn("rbob", joined)

    def test_spot_not_after_monday(self) -> None:
        self.assertTrue(self.df["spot_feature_timestamp"].le(self.df["retail_monday"]).all())
        self.assertTrue(self.df["spot_feature_timestamp"].lt(self.df["prediction_date"]).all())

    def test_inventory_lag_and_release(self) -> None:
        lag = (self.df["prediction_date"] - self.df["inventory_feature_timestamp"]).dt.days
        self.assertTrue((lag >= 10).all())
        self.assertTrue(self.df["inventory_release_ts_utc"].le(self.df["prediction_ts_utc"]).all())

    def test_target_is_next_week_only(self) -> None:
        self.assertTrue((self.df["target_timestamp"] - self.df["retail_monday"]).dt.days.eq(7).all())
        self.assertTrue(self.df["target_timestamp"].gt(self.df["prediction_date"]).all())

    def test_retail_momentum_not_target(self) -> None:
        self.assertFalse(self.df["retail_d7"].equals(self.df["target"]))
        self.assertTrue(self.df["retail_feature_timestamp"].eq(self.df["retail_monday"]).all())

    def test_prediction_is_tuesday_noon_et(self) -> None:
        self.assertTrue(self.df["prediction_date"].dt.day_name().eq("Tuesday").all())
        converted = self.df["prediction_ts_utc"].dt.tz_convert("America/New_York")
        self.assertTrue((converted.dt.hour == 12).all())

    def test_wpsr_release_is_wednesday_or_thursday(self) -> None:
        friday = pd.Timestamp("2024-08-09")
        holidays = pd.DatetimeIndex([])
        release = _wpsr_release_ts(friday, holidays)
        self.assertEqual(release.day_name(), "Wednesday")
        self.assertEqual(release.hour, 12)

    def test_assert_helper_on_kept_rows(self) -> None:
        _assert_no_leakage(self.df)

    def test_chronological_order(self) -> None:
        self.assertTrue(self.df["prediction_date"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
