"""Tests for the personal Gas Wait CLI."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from gas_wait_cli import (
    HISTORY_COLUMNS,
    append_signal_history,
    forecast_to_json,
    format_history_summary,
    format_signal_output,
    main,
)
from modeling.inference import SignalForecast, check_data_freshness
from modeling.signals import SignalResult
from modeling.weekly_pipeline import PROCESSED_DIR

EASTERN = ZoneInfo("America/New_York")


def _forecast(
    *,
    predicted: float = -0.034,
    signal: str = "WAIT",
    latest: str = "2026-08-17",
    warning: str | None = None,
) -> SignalForecast:
    row = pd.Series({"retail_d7": -0.01, "wti_d5": -0.2, "nyh_d5": -0.01})
    return SignalForecast(
        prediction_date=pd.Timestamp("2026-08-18"),
        retail_monday=pd.Timestamp("2026-08-17"),
        predicted_change=predicted,
        signal=SignalResult(signal=signal, predicted_change=predicted, threshold=0.03),  # type: ignore[arg-type]
        latest_data_date=pd.Timestamp(latest),
        spot_feature_timestamp=pd.Timestamp("2026-08-17"),
        inventory_feature_timestamp=pd.Timestamp("2026-08-08"),
        feature_row=row,
        explanations=("Recent retail momentum is negative", "WTI is falling"),
        data_warning=warning,
    )


class GasWaitCliTests(unittest.TestCase):
    def test_output_required_fields(self) -> None:
        text = format_signal_output(_forecast())
        self.assertIn("GAS WAIT", text)
        self.assertIn("Signal: WAIT", text)
        self.assertIn("Expected next-week move: -3.4¢ / gallon", text)
        self.assertIn("Horizon: ~7 days", text)
        self.assertIn("Last available data: 2026-08-17", text)
        self.assertIn("National weekly signal — not local station pricing.", text)
        self.assertNotIn("tomorrow", text.lower())
        self.assertNotIn("72 hour", text.lower())

    def test_stale_warning_displayed(self) -> None:
        warning = (
            "WARNING: Latest retail print is from 2026-08-10 "
            "(8 day(s) behind the expected Monday 2026-08-18). Signal may be stale."
        )
        text = format_signal_output(_forecast(warning=warning))
        self.assertIn("WARNING:", text)

    def test_signal_classification_fill(self) -> None:
        text = format_signal_output(_forecast(predicted=0.05, signal="FILL UP"))
        self.assertIn("Signal: FILL UP", text)
        self.assertIn("Expected next-week move: +5.0¢ / gallon", text)

    def test_json_output_structure(self) -> None:
        payload = forecast_to_json(_forecast(predicted=0.0366, signal="FILL UP"))
        self.assertEqual(payload["signal"], "FILL UP")
        self.assertAlmostEqual(payload["predicted_change_dollars_per_gallon"], 0.0366)
        self.assertEqual(payload["predicted_change_cents_per_gallon"], 3.7)
        self.assertEqual(payload["horizon"], "7 days")
        self.assertEqual(payload["latest_data_date"], "2026-08-17")
        self.assertEqual(payload["model"], "ridge_full")
        self.assertEqual(
            payload["explanations"],
            [
                "Recent pump prices have been moving down.",
                "Crude oil prices have been falling.",
            ],
        )

    def test_main_json_flag(self) -> None:
        fc = _forecast(predicted=0.0366, signal="FILL UP")
        buf = io.StringIO()
        with patch("gas_wait_cli.generate_weekly_signal", return_value=fc):
            with patch("gas_wait_cli.append_signal_history"):
                with patch("sys.stdout", buf):
                    code = main(["--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["signal"], "FILL UP")
        self.assertEqual(payload["model"], "ridge_full")

    def test_history_duplicate_prevention_and_preserve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hist = Path(tmpdir) / "signal_history.csv"
            fc1 = _forecast()
            fc2 = _forecast(
                predicted=0.05,
                signal="FILL UP",
            )
            fc2_other = SignalForecast(
                prediction_date=pd.Timestamp("2026-08-25"),
                retail_monday=pd.Timestamp("2026-08-24"),
                predicted_change=0.05,
                signal=SignalResult(signal="FILL UP", predicted_change=0.05, threshold=0.03),  # type: ignore[arg-type]
                latest_data_date=pd.Timestamp("2026-08-24"),
                spot_feature_timestamp=pd.Timestamp("2026-08-24"),
                inventory_feature_timestamp=pd.Timestamp("2026-08-15"),
                feature_row=pd.Series(),
                explanations=(),
            )
            append_signal_history(fc1, hist)
            append_signal_history(fc1, hist)
            append_signal_history(fc2_other, hist)
            df = pd.read_csv(hist)
            self.assertEqual(len(df), 2)
            self.assertListEqual(list(df.columns), HISTORY_COLUMNS)
            self.assertEqual(df.iloc[0]["prediction_date"], "2026-08-18")
            self.assertEqual(df.iloc[1]["prediction_date"], "2026-08-25")

    def test_missing_history_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.csv"
            text = format_history_summary(missing)
            self.assertIn("No signal history yet", text)

    def test_stale_data_detection(self) -> None:
        if not (PROCESSED_DIR / "retail_gasoline_regular_us_weekly.csv").exists():
            self.skipTest("processed data missing")
        from modeling.inference import build_latest_signal_row

        row = build_latest_signal_row(PROCESSED_DIR)
        pred = pd.Timestamp(row.iloc[0]["prediction_date"])
        before = datetime(pred.year, pred.month, pred.day, 11, 0, tzinfo=EASTERN)
        fresh = check_data_freshness(row, now=before)
        self.assertTrue(fresh.is_stale)

    def test_missing_data_main_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code = main(["--processed-dir", tmpdir])
            self.assertEqual(code, 1)

    def test_main_history_flag(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            code = main(["--history"])
        self.assertEqual(code, 0)
        self.assertIn("SIGNAL HISTORY", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
