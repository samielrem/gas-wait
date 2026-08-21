"""Fit first-experiment baselines and Ridge models. No hyperparameter search on test."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .build_features import (
    ALL_MODEL_FEATURES,
    INVENTORY_FEATURES,
    MARKET_FEATURES,
    RETAIL_FEATURES,
    SEASONALITY_FEATURES,
)

logger = logging.getLogger(__name__)

RIDGE_ALPHA = 1.0
TRAIN_FRACTION = 0.70

FEATURE_SETS = {
    "momentum_baseline": RETAIL_FEATURES[:1],
    "ridge_retail": RETAIL_FEATURES,
    "ridge_retail_market": RETAIL_FEATURES + MARKET_FEATURES,
    "ridge_full": ALL_MODEL_FEATURES,
}


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp


@dataclass
class FitResult:
    name: str
    predictions: pd.Series
    fitted: object | None
    features: list[str]


def chronological_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> Split:
    ordered = df.sort_values("prediction_date").reset_index(drop=True)
    cutoff_idx = int(len(ordered) * train_fraction)
    cutoff_idx = max(1, min(cutoff_idx, len(ordered) - 1))
    cutoff = ordered.loc[cutoff_idx, "prediction_date"]
    train = ordered.loc[ordered["prediction_date"] < cutoff].copy()
    test = ordered.loc[ordered["prediction_date"] >= cutoff].copy()
    logger.info(
        "Chronological split cutoff=%s train=%s test=%s",
        cutoff.date(),
        len(train),
        len(test),
    )
    return Split(train=train, test=test, cutoff=cutoff)


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def fit_predict_holdout(split: Split) -> dict[str, FitResult]:
    results: dict[str, FitResult] = {}
    y_train = split.train["target"].to_numpy()
    test_index = split.test.index

    results["momentum_baseline"] = FitResult(
        name="momentum_baseline",
        predictions=split.test["retail_d7"].copy(),
        fitted=None,
        features=["retail_d7"],
    )

    for name, features in FEATURE_SETS.items():
        if name == "momentum_baseline":
            continue
        model = _ridge()
        model.fit(split.train[features], y_train)
        pred = pd.Series(model.predict(split.test[features]), index=test_index)
        results[name] = FitResult(name=name, predictions=pred, fitted=model, features=features)
        logger.info("Fitted %s on %s rows, %s features", name, len(split.train), len(features))
    return results


def walk_forward_predict(df: pd.DataFrame, features: list[str], min_train: int) -> pd.Series:
    """Expanding yearly walk-forward. Retrain on all weeks before each year."""
    ordered = df.sort_values("prediction_date").copy()
    preds = pd.Series(index=ordered["prediction_date"], dtype=float, name="prediction")
    years = sorted(ordered["prediction_date"].dt.year.unique())
    for year in years:
        train = ordered.loc[ordered["prediction_date"].dt.year < year]
        test = ordered.loc[ordered["prediction_date"].dt.year == year]
        if len(train) < min_train or test.empty:
            continue
        if features == ["retail_d7"]:
            preds.loc[test["prediction_date"]] = test["retail_d7"].to_numpy()
            continue
        model = _ridge()
        model.fit(train[features], train["target"])
        preds.loc[test["prediction_date"]] = model.predict(test[features])
    return preds


def walk_forward_all(df: pd.DataFrame, min_train: int | None = None) -> dict[str, pd.Series]:
    if min_train is None:
        min_train = max(52 * 8, int(len(df) * 0.40))
    out: dict[str, pd.Series] = {}
    ordered = df.sort_values("prediction_date").reset_index(drop=True)
    out["momentum_baseline"] = walk_forward_predict(ordered, ["retail_d7"], min_train)
    for name, features in FEATURE_SETS.items():
        if name == "momentum_baseline":
            continue
        out[name] = walk_forward_predict(ordered, features, min_train)
    return out
