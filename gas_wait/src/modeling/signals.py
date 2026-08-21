"""WAIT / FILL UP / NO CLEAR SIGNAL decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .config import DEFAULT_SIGNAL_THRESHOLD

SignalLabel = Literal["WAIT", "FILL UP", "NO CLEAR SIGNAL"]


@dataclass(frozen=True)
class SignalResult:
    signal: SignalLabel
    predicted_change: float
    threshold: float
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "predicted_change": self.predicted_change,
            "threshold": self.threshold,
            "confidence": self.confidence,
        }


def generate_signal(
    predicted_change: float,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    confidence: float | None = None,
) -> SignalResult:
    """Map a predicted price change ($/gal) to a consumer signal.

    Research defaults (not optimized on test):
    - WAIT if predicted_change <= -threshold
    - FILL UP if predicted_change >= +threshold
    - otherwise NO CLEAR SIGNAL
    """
    if predicted_change <= -threshold:
        label: SignalLabel = "WAIT"
    elif predicted_change >= threshold:
        label = "FILL UP"
    else:
        label = "NO CLEAR SIGNAL"
    return SignalResult(
        signal=label,
        predicted_change=float(predicted_change),
        threshold=float(threshold),
        confidence=confidence,
    )


def generate_signals_series(
    predictions: pd.Series,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
) -> pd.Series:
    """Vectorized signal labels for a prediction series."""
    out = pd.Series("NO CLEAR SIGNAL", index=predictions.index, dtype=object)
    out.loc[predictions <= -threshold] = "WAIT"
    out.loc[predictions >= threshold] = "FILL UP"
    return out
