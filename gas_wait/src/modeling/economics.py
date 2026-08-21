"""Theoretical economic evaluation for Gas Wait signals.

All dollar amounts are **benchmarks** assuming a fixed tank size at the
observed average price series. They do not claim real consumer fill behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_TANK_GALLONS
from .signals import generate_signals_series


@dataclass(frozen=True)
class EconomicsSummary:
    n: int
    n_wait: int
    n_fill: int
    n_none: int
    pct_silent: float
    pct_correct_among_signals: float
    mean_actual_after_wait: float | None
    mean_actual_after_fill: float | None
    mean_savings_per_period_usd: float
    total_savings_usd: float
    cumulative_savings_usd: float


def wait_savings_per_period(actual_change: float | pd.Series, gallons: float) -> float | pd.Series:
    """Theoretical savings vs fill-now when WAIT is chosen.

    If price falls next period (actual_change < 0), waiting saves money.
    ``savings = -gallons * actual_change`` ($/gal change × gallons).
    """
    return -gallons * actual_change


def evaluate_signals(
    actual_change: pd.Series,
    predicted_change: pd.Series,
    *,
    threshold: float = 0.03,
    gallons: float = DEFAULT_TANK_GALLONS,
) -> EconomicsSummary:
    """Summarize theoretical economics for predicted WAIT/FILL signals."""
    mask = actual_change.notna() & predicted_change.notna()
    actual = actual_change.loc[mask]
    pred = predicted_change.loc[mask]
    action = generate_signals_series(pred, threshold)
    wait = action == "WAIT"
    fill = action == "FILL UP"
    none = action == "NO CLEAR SIGNAL"
    signaled = wait | fill

    wait_correct = (actual < 0) & wait
    fill_correct = (actual > 0) & fill
    n_signaled = int(signaled.sum())
    n_correct = int((wait_correct | fill_correct).sum())
    pct_correct = n_correct / n_signaled if n_signaled else float("nan")

    pnl = pd.Series(0.0, index=actual.index)
    pnl.loc[wait] = wait_savings_per_period(actual.loc[wait], gallons)
    # FILL UP and NO SIGNAL contribute 0 vs always-fill-now in this benchmark.

    return EconomicsSummary(
        n=len(actual),
        n_wait=int(wait.sum()),
        n_fill=int(fill.sum()),
        n_none=int(none.sum()),
        pct_silent=float(none.mean()) if len(none) else float("nan"),
        pct_correct_among_signals=float(pct_correct),
        mean_actual_after_wait=float(actual.loc[wait].mean()) if wait.any() else None,
        mean_actual_after_fill=float(actual.loc[fill].mean()) if fill.any() else None,
        mean_savings_per_period_usd=float(pnl.mean()),
        total_savings_usd=float(pnl.sum()),
        cumulative_savings_usd=float(pnl.sum()),
    )


def cumulative_savings_series(
    actual_change: pd.Series,
    predicted_change: pd.Series,
    *,
    threshold: float = 0.03,
    gallons: float = DEFAULT_TANK_GALLONS,
) -> pd.Series:
    """Period-by-period cumulative theoretical savings (WAIT-only P&L)."""
    mask = actual_change.notna() & predicted_change.notna()
    actual = actual_change.loc[mask]
    pred = predicted_change.loc[mask]
    action = generate_signals_series(pred, threshold)
    pnl = pd.Series(0.0, index=actual.index)
    wait = action == "WAIT"
    pnl.loc[wait] = [-gallons * a for a in actual.loc[wait]]
    return pnl.cumsum()
