"""Point-in-time and leakage assertion tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from modeling.point_in_time import (
    PublicationRule,
    assert_feature_timestamps,
    attach_public_availability,
    daily_spot_public_ts,
    filter_known_at,
    wpsr_release_ts,
)


class PointInTimeTests(unittest.TestCase):
    def test_daily_spot_lags_one_business_day(self) -> None:
        session = pd.Timestamp("2024-08-16")  # Friday
        pub = daily_spot_public_ts(session)
        self.assertEqual(pub.day_name(), "Monday")
        self.assertEqual(pub.hour, 12)

    def test_filter_known_at_excludes_future(self) -> None:
        df = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
                "value": [1.0, 2.0],
            }
        )
        df = attach_public_availability(df, PublicationRule.WEEKLY_RETAIL_TUESDAY_NOON_ET)
        pred = pd.Timestamp("2024-01-02 12:00", tz="America/New_York")
        known = filter_known_at(df, pred)
        self.assertEqual(len(known), 1)
        self.assertEqual(known.iloc[0]["observation_date"], pd.Timestamp("2024-01-01"))

    def test_assert_feature_timestamps_fails_on_leak(self) -> None:
        df = pd.DataFrame(
            {
                "prediction_ts_utc": pd.to_datetime(["2024-01-09"]),
                "spot_feature_timestamp": pd.to_datetime(["2024-01-10"]),
            }
        )
        with self.assertRaises(AssertionError):
            assert_feature_timestamps(
                df,
                "prediction_ts_utc",
                {"spot": "spot_feature_timestamp"},
            )

    def test_future_row_never_passes_filter(self) -> None:
        """Regression: data published after prediction must not appear."""
        right = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
                "value": [70.0, 71.0],
            }
        )
        right = attach_public_availability(right, PublicationRule.DAILY_SPOT_NEXT_WEEKDAY_NOON_ET)
        pred = pd.Timestamp("2024-01-05 11:00", tz="America/New_York")
        known = filter_known_at(right, pred, availability_col="public_available_ts")
        self.assertEqual(len(known), 0)

    def test_wpsr_release_wednesday(self) -> None:
        friday = pd.Timestamp("2024-08-09")
        release = wpsr_release_ts(friday, pd.DatetimeIndex([]))
        self.assertEqual(release.day_name(), "Wednesday")


if __name__ == "__main__":
    unittest.main()
