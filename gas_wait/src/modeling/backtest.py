"""Walk-forward and holdout backtesting for Gas Wait models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from .config import DEFAULT_SIGNAL_THRESHOLD, FeatureGroup, resolve_feature_columns
from .economics import evaluate_signals
from .signals import generate_signals_series


class TrainMode(str, Enum):
    HOLDOUT = "holdout"
    EXPANDING = "expanding"
    ROLLING = "rolling"


class RetrainFrequency(str, Enum):
    EACH_ROW = "each_row"
    YEARLY = "yearly"
    NEVER = "never"


class Predictor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class BacktestConfig:
    train_mode: TrainMode = TrainMode.HOLDOUT
    train_fraction: float = 0.70
    rolling_window: int | None = None
    retrain_frequency: RetrainFrequency | int = RetrainFrequency.YEARLY
    min_train_rows: int = 52
    target_col: str = "target"
    time_col: str = "prediction_date"
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD
    feature_groups: list[FeatureGroup] | list[str] = field(default_factory=lambda: list(FeatureGroup))


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    economics: Any
    config: BacktestConfig

    @property
    def is_chronological(self) -> bool:
        ts = self.predictions[self.config.time_col]
        return bool(ts.is_monotonic_increasing)


def chronological_split(
    df: pd.DataFrame,
    time_col: str,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ordered = df.sort_values(time_col).reset_index(drop=True)
    idx = int(len(ordered) * train_fraction)
    idx = max(1, min(idx, len(ordered) - 1))
    cutoff = ordered.loc[idx, time_col]
    train = ordered.loc[ordered[time_col] < cutoff].copy()
    test = ordered.loc[ordered[time_col] >= cutoff].copy()
    return train, test, cutoff


def _retrain_mask(train: pd.DataFrame, test_row: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    if config.train_mode is TrainMode.EXPANDING:
        pool = train.loc[train[config.time_col] < test_row[config.time_col]]
    elif config.train_mode is TrainMode.ROLLING:
        if config.rolling_window is None:
            raise ValueError("rolling_window required for ROLLING mode")
        pool = train.loc[train[config.time_col] < test_row[config.time_col]].tail(config.rolling_window)
    else:
        raise ValueError("Use run_holdout_backtest for HOLDOUT mode")
    return pool


def run_holdout_backtest(
    dataset: pd.DataFrame,
    model: Predictor,
    config: BacktestConfig | None = None,
    *,
    feature_cols: list[str] | None = None,
) -> BacktestResult:
    """Single chronological train/test split (exp01-style 70/30)."""
    config = config or BacktestConfig()
    if feature_cols is None:
        groups = config.feature_groups
        if groups and isinstance(groups[0], FeatureGroup):
            feature_cols = resolve_feature_columns(groups)  # type: ignore[arg-type]
        else:
            feature_cols = resolve_feature_columns(groups)  # type: ignore[arg-type]

    train, test, _ = chronological_split(dataset, config.time_col, config.train_fraction)
    model.fit(train[feature_cols], train[config.target_col])
    pred = model.predict(test[feature_cols])
    out = test[[config.time_col, config.target_col]].copy()
    out["predicted"] = pred
    out["error"] = out["predicted"] - out[config.target_col]
    out["direction_correct"] = np.sign(out["predicted"]) == np.sign(out[config.target_col])
    out["signal"] = generate_signals_series(out["predicted"], config.signal_threshold)
    econ = evaluate_signals(out[config.target_col], out["predicted"], threshold=config.signal_threshold)
    return BacktestResult(predictions=out, economics=econ, config=config)


def run_walk_forward_backtest(
    dataset: pd.DataFrame,
    model_factory: Callable[[], Predictor],
    config: BacktestConfig | None = None,
    *,
    feature_cols: list[str] | None = None,
) -> BacktestResult:
    """Expanding or rolling walk-forward with configurable retrain frequency."""
    config = config or BacktestConfig(train_mode=TrainMode.EXPANDING)
    if config.train_mode is TrainMode.HOLDOUT:
        raise ValueError("Use run_holdout_backtest for HOLDOUT mode")

    if feature_cols is None:
        feature_cols = resolve_feature_columns(config.feature_groups)  # type: ignore[arg-type]

    ordered = dataset.sort_values(config.time_col).reset_index(drop=True)
    preds: list[dict] = []
    cached_model: Predictor | None = None
    last_fit_year: int | None = None

    for i, row in ordered.iterrows():
        train = _retrain_mask(ordered, row, config)
        if len(train) < config.min_train_rows:
            continue

        year = pd.Timestamp(row[config.time_col]).year
        do_fit = False
        if config.retrain_frequency is RetrainFrequency.NEVER:
            do_fit = cached_model is None
        elif config.retrain_frequency is RetrainFrequency.EACH_ROW:
            do_fit = True
        elif config.retrain_frequency is RetrainFrequency.YEARLY:
            do_fit = last_fit_year != year
        elif isinstance(config.retrain_frequency, int):
            do_fit = i % config.retrain_frequency == 0
        else:
            do_fit = True

        if do_fit or cached_model is None:
            cached_model = model_factory()
            cached_model.fit(train[feature_cols], train[config.target_col])
            last_fit_year = year

        pred = float(cached_model.predict(row[feature_cols].to_frame().T)[0])
        actual = float(row[config.target_col])
        preds.append(
            {
                config.time_col: row[config.time_col],
                config.target_col: actual,
                "predicted": pred,
                "error": pred - actual,
                "direction_correct": np.sign(pred) == np.sign(actual),
                "signal": generate_signals_series(pd.Series([pred]), config.signal_threshold).iloc[0],
            }
        )

    out = pd.DataFrame(preds)
    econ = evaluate_signals(out[config.target_col], out["predicted"], threshold=config.signal_threshold)
    return BacktestResult(predictions=out, economics=econ, config=config)


def run_simple_backtest(
    dataset: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], pd.Series],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest without training — e.g. momentum baseline ``predict_fn(row) -> retail_d7``."""
    config = config or BacktestConfig()
    ordered = dataset.sort_values(config.time_col).reset_index(drop=True)
    pred = predict_fn(ordered)
    out = ordered[[config.time_col, config.target_col]].copy()
    out["predicted"] = pred.values
    out["error"] = out["predicted"] - out[config.target_col]
    out["direction_correct"] = np.sign(out["predicted"]) == np.sign(out[config.target_col])
    out["signal"] = generate_signals_series(out["predicted"], config.signal_threshold)
    econ = evaluate_signals(out[config.target_col], out["predicted"], threshold=config.signal_threshold)
    return BacktestResult(predictions=out, economics=econ, config=config)
