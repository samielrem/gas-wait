"""Target construction interface for Gas Wait modeling.

Daily horizons (24h / 48h / 72h) require **actual daily retail observations**.
They are declared here but not implemented on weekly EIA data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class TargetKind(str, Enum):
    """Supported prediction horizons."""

    H24 = "24h"
    H48 = "48h"
    H72 = "72h"
    H7D = "7d"
    NEXT_WEEK_RETAIL_CHANGE = "next_week_retail_change"


@dataclass(frozen=True)
class TargetSpec:
    kind: TargetKind
    price_col: str = "retail_price"
    date_col: str = "observation_date"
    frequency: str = "weekly"  # "weekly" | "daily"


DAILY_TARGET_MESSAGE = (
    "Target {kind} requires licensed daily retail gasoline observations. "
    "Do not synthesize daily targets from weekly EIA retail."
)


def _require_daily(spec: TargetSpec) -> None:
    if spec.frequency != "daily":
        raise NotImplementedError(DAILY_TARGET_MESSAGE.format(kind=spec.kind.value))


def build_target_column(
    retail: pd.DataFrame,
    spec: TargetSpec,
) -> pd.DataFrame:
    """Attach ``target`` and ``target_timestamp`` for the requested horizon.

    Only ``NEXT_WEEK_RETAIL_CHANGE`` on weekly EIA retail is implemented.
    """
    out = retail.sort_values(spec.date_col).drop_duplicates(spec.date_col).copy()
    if spec.kind is TargetKind.NEXT_WEEK_RETAIL_CHANGE:
        if spec.frequency != "weekly":
            raise ValueError("next_week_retail_change expects weekly retail frequency.")
        out["next_date"] = out[spec.date_col].shift(-1)
        out["next_price"] = out[spec.price_col].shift(-1)
        out["gap_days"] = (out["next_date"] - out[spec.date_col]).dt.days
        out["target_timestamp"] = out["next_date"]
        out["target"] = out["next_price"] - out[spec.price_col]
        out["target_valid"] = out["gap_days"].eq(7) & out["next_price"].notna()
        return out

    if spec.kind in (TargetKind.H24, TargetKind.H48, TargetKind.H72, TargetKind.H7D):
        _require_daily(spec)

    raise NotImplementedError(f"Target {spec.kind} is not implemented.")


def attach_weekly_prediction_clock(
    labeled: pd.DataFrame,
    *,
    retail_date_col: str = "observation_date",
) -> pd.DataFrame:
    """Add Tuesday-noon prediction timestamps for weekly EIA rows (exp01 clock)."""
    from .point_in_time import _at_noon_et, _to_utc, weekly_retail_public_ts

    out = labeled.copy()
    out["retail_monday"] = out[retail_date_col]
    out["prediction_date"] = out["retail_monday"] + pd.Timedelta(days=1)
    out["prediction_ts_et"] = out["prediction_date"].map(_at_noon_et)
    out["prediction_ts_utc"] = _to_utc(out["prediction_ts_et"])
    out["retail_public_ts"] = out["retail_monday"].map(weekly_retail_public_ts)
    out["retail_feature_timestamp"] = out["retail_monday"]
    return out


def drop_invalid_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows with invalid targets (non-weekly gap, missing next price)."""
    if "target_valid" not in df.columns:
        return df, pd.DataFrame()
    kept = df.loc[df["target_valid"]].copy()
    dropped = df.loc[~df["target_valid"]].copy()
    return kept, dropped
