"""Signal and economics tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from modeling.economics import evaluate_signals, wait_savings_per_period
from modeling.signals import generate_signal


class SignalEconomicsTests(unittest.TestCase):
    def test_signal_thresholds(self) -> None:
        self.assertEqual(generate_signal(-0.04).signal, "WAIT")
        self.assertEqual(generate_signal(0.04).signal, "FILL UP")
        self.assertEqual(generate_signal(0.01).signal, "NO CLEAR SIGNAL")

    def test_wait_savings_when_price_falls(self) -> None:
        self.assertAlmostEqual(wait_savings_per_period(-0.10, 15), 1.5)

    def test_economics_counts(self) -> None:
        actual = pd.Series([0.05, -0.05, 0.01])
        pred = pd.Series([0.05, -0.05, 0.01])
        summary = evaluate_signals(actual, pred, threshold=0.03)
        self.assertEqual(summary.n_wait, 1)
        self.assertEqual(summary.n_fill, 1)
        self.assertEqual(summary.n_none, 1)


if __name__ == "__main__":
    unittest.main()
