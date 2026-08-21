"""Target construction tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from modeling.targets import TargetKind, TargetSpec, build_target_column


class TargetTests(unittest.TestCase):
    def test_weekly_next_week_target(self) -> None:
        df = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
                "retail_price": [3.0, 3.1, 3.05],
            }
        )
        out = build_target_column(df, TargetSpec(kind=TargetKind.NEXT_WEEK_RETAIL_CHANGE))
        self.assertAlmostEqual(out.loc[0, "target"], 0.1)
        self.assertTrue(out.loc[0, "target_valid"])
        self.assertFalse(out.loc[2, "target_valid"])

    def test_daily_target_raises(self) -> None:
        df = pd.DataFrame({"observation_date": pd.to_datetime(["2024-01-01"]), "retail_price": [3.0]})
        spec = TargetSpec(kind=TargetKind.H72, frequency="weekly")
        with self.assertRaises(NotImplementedError):
            build_target_column(df, spec)


if __name__ == "__main__":
    unittest.main()
