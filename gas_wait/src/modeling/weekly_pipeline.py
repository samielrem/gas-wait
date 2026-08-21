"""Build weekly EIA modeling rows using the reusable modeling framework.

This module orchestrates ``features``, ``point_in_time``, and ``targets`` for
the exp01 weekly experiment. It does **not** write ``weekly_model_dataset.csv``
unless explicitly requested — validation uses it read-only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FeatureGroup, resolve_feature_columns
from .features import (
    add_crude_features,
    add_inventory_level_features,
    add_retail_momentum_weekly,
    add_seasonality_features,
    add_wholesale_features,
)
from .point_in_time import (
    PublicationRule,
    _to_utc,
    _us_federal_holidays,
    asof_on_dates,
    attach_public_availability,
    assert_feature_timestamps,
    assert_no_future_observations,
)
from .targets import TargetKind, TargetSpec, attach_weekly_prediction_clock, build_target_column, drop_invalid_targets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _load_series(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    out = df[["observation_date", "value"]].rename(columns={"value": value_name})
    return out.sort_values("observation_date").drop_duplicates("observation_date").reset_index(drop=True)


def build_weekly_eia_dataset(
    processed_dir: Path = PROCESSED_DIR,
    *,
    require_target: bool = True,
) -> pd.DataFrame:
    """Construct weekly modeling rows from raw EIA processed CSVs via the framework.

    When ``require_target`` is False, rows without a published next-week retail
    target are retained (used for the live CLI signal on the latest Monday).
    """
    retail = _load_series(processed_dir / "retail_gasoline_regular_us_weekly.csv", "retail_price")
    wti = _load_series(processed_dir / "wti_crude_spot_us_daily.csv", "wti")
    nyh = _load_series(processed_dir / "ny_harbor_gasoline_regular_spot_daily.csv", "nyh")
    gc = _load_series(processed_dir / "gulf_coast_gasoline_regular_spot_daily.csv", "gc")
    inv = _load_series(processed_dir / "gasoline_inventories_us_weekly.csv", "stocks")

    labeled = build_target_column(
        retail,
        TargetSpec(kind=TargetKind.NEXT_WEEK_RETAIL_CHANGE, price_col="retail_price"),
    )
    if require_target:
        labeled, _ = drop_invalid_targets(labeled)
    frame = attach_weekly_prediction_clock(labeled)
    frame = add_retail_momentum_weekly(frame, "retail_price", date_col="retail_monday")

    wti_f = add_crude_features(wti, "wti", include_extended=False)
    nyh_f = add_wholesale_features(nyh, "nyh", "nyh", include_extended=False)
    gc_f = add_wholesale_features(gc, "gc", "gc", include_extended=False)

    wti_f = attach_public_availability(wti_f, PublicationRule.DAILY_SPOT_NEXT_WEEKDAY_NOON_ET)
    nyh_f = attach_public_availability(nyh_f, PublicationRule.DAILY_SPOT_NEXT_WEEKDAY_NOON_ET)
    gc_f = attach_public_availability(gc_f, PublicationRule.DAILY_SPOT_NEXT_WEEKDAY_NOON_ET)

    holidays = _us_federal_holidays(
        inv["observation_date"].min() - pd.Timedelta(days=14),
        frame["prediction_date"].max() + pd.Timedelta(days=14),
    )
    inv_f = add_inventory_level_features(inv, value_col="stocks")
    inv_f = attach_public_availability(
        inv_f,
        PublicationRule.WPSR_WEDNESDAY_NOON_ET,
        holidays=holidays,
    )
    inv_f["inventory_feature_timestamp"] = inv_f["observation_date"]
    inv_f["inventory_release_ts_utc"] = inv_f["public_available_ts_utc"]

    spot_cols_wti = ["wti_d1", "wti_d3", "wti_d5", "wti_vol_20", "wti_chg_7cal"]
    spot_cols_nyh = ["nyh_d1", "nyh_d5", "nyh_vol_20"]
    spot_cols_gc = ["gc_d1", "gc_d5"]

    # As-of Monday T (strictly before Tuesday prediction clock; matches exp01).
    wti_j = asof_on_dates(frame["retail_monday"], wti_f, spot_cols_wti, out_obs_col="spot_feature_timestamp")
    nyh_j = asof_on_dates(frame["retail_monday"], nyh_f, spot_cols_nyh, out_obs_col="nyh_timestamp")
    gc_j = asof_on_dates(frame["retail_monday"], gc_f, spot_cols_gc, out_obs_col="gc_timestamp")

    for col in spot_cols_wti:
        frame[col] = wti_j[col].to_numpy()
    frame["spot_feature_timestamp"] = wti_j["spot_feature_timestamp"].to_numpy()
    for col in spot_cols_nyh:
        frame[col] = nyh_j[col].to_numpy()
    for col in spot_cols_gc:
        frame[col] = gc_j[col].to_numpy()
    frame["crack_nyh_d5"] = 42.0 * frame["nyh_d5"] - frame["wti_d5"]

    inv_j = pd.merge_asof(
        frame[["retail_monday", "prediction_ts_utc"]].sort_values("prediction_ts_utc"),
        inv_f[
            ["inventory_release_ts_utc", "inventory_feature_timestamp", "inv_wow", "inv_seasonal_z"]
        ].sort_values("inventory_release_ts_utc"),
        left_on="prediction_ts_utc",
        right_on="inventory_release_ts_utc",
        direction="backward",
    )
    frame = frame.merge(
        inv_j[["retail_monday", "inventory_feature_timestamp", "inventory_release_ts_utc", "inv_wow", "inv_seasonal_z"]],
        on="retail_monday",
        how="left",
    )

    seas = add_seasonality_features(frame["prediction_date"])
    frame = frame.merge(seas, on="prediction_date", how="left")
    frame["retail_last"] = frame["retail_price"]
    frame["target_timestamp"] = frame["next_date"]

    feature_cols = resolve_feature_columns(list(FeatureGroup))
    required = feature_cols + ["spot_feature_timestamp", "inventory_feature_timestamp"]
    if require_target:
        required = required + ["target"]
    frame = frame.dropna(subset=required).copy()

    assert_no_future_observations(
        frame,
        "prediction_date",
        ["retail_feature_timestamp", "spot_feature_timestamp", "inventory_feature_timestamp"],
    )
    assert_feature_timestamps(
        frame,
        "prediction_ts_utc",
        {
            "retail": "retail_feature_timestamp",
            "spot": "spot_feature_timestamp",
            "inventory": "inventory_feature_timestamp",
        },
    )
    pred_naive = frame["prediction_date"]
    if require_target:
        if (frame["target_timestamp"] <= pred_naive).any():
            raise AssertionError("Target timestamp must be after prediction_date")
    if not (frame["spot_feature_timestamp"] < pred_naive).all():
        raise AssertionError("Spot feature must be before prediction_date")

    keep = [
        "prediction_date",
        "prediction_ts_utc",
        "target",
        "target_timestamp",
        "retail_monday",
        "retail_feature_timestamp",
        "spot_feature_timestamp",
        "inventory_feature_timestamp",
        "inventory_release_ts_utc",
        *feature_cols,
        "retail_last",
        "wti_chg_7cal",
        "month",
    ]
    return frame[keep].sort_values("prediction_date").reset_index(drop=True)
