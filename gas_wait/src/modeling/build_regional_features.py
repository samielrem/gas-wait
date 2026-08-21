"""Point-in-time weekly features for one regional retail series.

Same Tuesday 12:00 ET clock as exp01. Does not write weekly_model_dataset.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .build_features import (
    DatasetBuildResult,
    PROCESSED_DIR,
    WTI_PATH,
    INV_PATH,
    _add_session_features,
    _asof_daily,
    _assert_no_leakage,
    _at_noon_et,
    _holiday_travel_week,
    _is_summer,
    _load_series,
    _to_utc,
    _us_federal_holidays,
    _wpsr_release_ts,
)
from .regional_config import RegionalExperiment, hub_csv_name, retail_csv_name

logger = logging.getLogger(__name__)

RETAIL_FEATURES = ["retail_d7", "retail_d14"]
WTI_FEATURES = ["wti_d1", "wti_d3", "wti_d5", "wti_vol_20"]
HUB_FEATURES = ["hub_d1", "hub_d5", "hub_vol_20", "crack_hub_d5"]
MARKET_FEATURES = WTI_FEATURES + HUB_FEATURES
INVENTORY_FEATURES = ["inv_wow", "inv_seasonal_z"]
SEASONALITY_FEATURES = ["sin_doy", "cos_doy", "is_summer"]
ALL_MODEL_FEATURES = RETAIL_FEATURES + MARKET_FEATURES + INVENTORY_FEATURES + SEASONALITY_FEATURES
MISMATCH_FEATURES = ["mis_d1", "mis_d5", "mis_vol_20"]
DIAGNOSTIC_HUB_FEATURES = ["nyh_d5", "gc_d5", "la_d5"]

FEATURE_SETS = {
    "momentum_baseline": ["retail_d7"],
    "ridge_retail": RETAIL_FEATURES,
    "ridge_matched": RETAIL_FEATURES + MARKET_FEATURES,
    "ridge_mismatched": RETAIL_FEATURES + WTI_FEATURES + MISMATCH_FEATURES,
    "ridge_full": ALL_MODEL_FEATURES,
}

TIMESTAMP_COLS = [
    "region_id",
    "prediction_date",
    "prediction_ts_utc",
    "target",
    "target_timestamp",
    "retail_monday",
    "retail_feature_timestamp",
    "spot_feature_timestamp",
    "inventory_feature_timestamp",
    "inventory_release_ts_utc",
]


def build_regional_dataset(
    region: RegionalExperiment,
    processed_dir: Path = PROCESSED_DIR,
) -> DatasetBuildResult:
    retail = _load_series(processed_dir / retail_csv_name(region), "retail_price")
    wti = _load_series(processed_dir / WTI_PATH.name, "wti")
    hub = _load_series(processed_dir / hub_csv_name(region.matched_hub), "hub")
    mis = _load_series(processed_dir / hub_csv_name(region.mismatched_hub), "mis")
    nyh = _load_series(processed_dir / "ny_harbor_gasoline_regular_spot_daily.csv", "nyh")
    gc = _load_series(processed_dir / "gulf_coast_gasoline_regular_spot_daily.csv", "gc")
    la = _load_series(processed_dir / "la_rbob_gasoline_regular_spot_daily.csv", "la")
    inv = _load_series(processed_dir / INV_PATH.name, "stocks")

    drop_records: list[dict] = []
    retail_valid = retail.dropna(subset=["retail_price"]).copy()
    for _, row in retail.loc[retail["retail_price"].isna()].iterrows():
        drop_records.append({"retail_monday": row["observation_date"], "reason": "missing_retail_at_t"})

    retail_valid["next_date"] = retail_valid["observation_date"].shift(-1)
    retail_valid["next_price"] = retail_valid["retail_price"].shift(-1)
    retail_valid["gap_days"] = (retail_valid["next_date"] - retail_valid["observation_date"]).dt.days
    bad_next = retail_valid["gap_days"].ne(7) | retail_valid["next_price"].isna()
    for _, row in retail_valid.loc[bad_next].iterrows():
        drop_records.append(
            {"retail_monday": row["observation_date"], "reason": "missing_or_nonweekly_retail_at_tplus7"}
        )
    candidates = retail_valid.loc[~bad_next].copy()
    candidates["retail_monday"] = candidates["observation_date"]
    candidates["prediction_date"] = candidates["retail_monday"] + pd.Timedelta(days=1)
    candidates["prediction_ts_et"] = candidates["prediction_date"].map(_at_noon_et)
    candidates["prediction_ts_utc"] = _to_utc(candidates["prediction_ts_et"])
    candidates["target_timestamp"] = candidates["next_date"]
    candidates["target"] = candidates["next_price"] - candidates["retail_price"]
    candidates["retail_last"] = candidates["retail_price"]
    candidates["retail_feature_timestamp"] = candidates["retail_monday"]
    candidates["retail_d7"] = candidates["retail_price"] - candidates["retail_price"].shift(1)
    prev2 = candidates["retail_price"].shift(2)
    prev1_date = candidates["retail_monday"].shift(1)
    prev2_date = candidates["retail_monday"].shift(2)
    week_ok = (
        (candidates["retail_monday"] - prev1_date).dt.days.eq(7)
        & (candidates["retail_monday"] - prev2_date).dt.days.eq(14)
    )
    candidates["retail_d14"] = np.where(week_ok, candidates["retail_price"] - prev2, np.nan)

    wti_f = _add_session_features(wti, "wti", "wti")
    hub_f = _add_session_features(hub, "hub", "hub")
    mis_f = _add_session_features(mis, "mis", "mis")
    nyh_f = _add_session_features(nyh, "nyh", "nyh")
    gc_f = _add_session_features(gc, "gc", "gc")
    la_f = _add_session_features(la, "la", "la")

    wti_join = _asof_daily(
        candidates["retail_monday"],
        wti_f,
        ["wti_d1", "wti_d3", "wti_d5", "wti_vol_20"],
    )
    hub_join = _asof_daily(
        candidates["retail_monday"],
        hub_f,
        ["hub_d1", "hub_d5", "hub_vol_20"],
    ).rename(columns={"spot_feature_timestamp": "hub_timestamp"})
    mis_join = _asof_daily(
        candidates["retail_monday"],
        mis_f,
        ["mis_d1", "mis_d5", "mis_vol_20"],
    ).rename(columns={"spot_feature_timestamp": "mis_timestamp"})
    nyh_join = _asof_daily(candidates["retail_monday"], nyh_f, ["nyh_d5"]).rename(
        columns={"spot_feature_timestamp": "nyh_timestamp"}
    )
    gc_join = _asof_daily(candidates["retail_monday"], gc_f, ["gc_d5"]).rename(
        columns={"spot_feature_timestamp": "gc_timestamp"}
    )
    la_join = _asof_daily(candidates["retail_monday"], la_f, ["la_d5"]).rename(
        columns={"spot_feature_timestamp": "la_timestamp"}
    )

    frame = candidates.merge(wti_join, on="retail_monday", how="left")
    frame = frame.merge(hub_join, on="retail_monday", how="left")
    frame = frame.merge(mis_join, on="retail_monday", how="left")
    frame = frame.merge(nyh_join, on="retail_monday", how="left")
    frame = frame.merge(gc_join, on="retail_monday", how="left")
    frame = frame.merge(la_join, on="retail_monday", how="left")
    frame["spot_feature_timestamp"] = frame[["spot_feature_timestamp", "hub_timestamp"]].max(axis=1)
    frame["crack_hub_d5"] = 42.0 * frame["hub_d5"] - frame["wti_d5"]
    frame["region_id"] = region.region_id

    holidays = _us_federal_holidays(
        inv["observation_date"].min() - pd.Timedelta(days=14),
        candidates["prediction_date"].max() + pd.Timedelta(days=14),
    )
    inv = inv.dropna(subset=["stocks"]).copy()
    inv["inventory_feature_timestamp"] = inv["observation_date"]
    inv["inventory_release_ts_et"] = inv["observation_date"].map(lambda d: _wpsr_release_ts(d, holidays))
    inv["inventory_release_ts_utc"] = _to_utc(inv["inventory_release_ts_et"])
    inv["inv_wow"] = inv["stocks"].diff(1)
    inv["weekofyear"] = inv["observation_date"].dt.isocalendar().week.astype(int)
    inv["year"] = inv["observation_date"].dt.year
    zscores: list[float] = []
    for _, row in inv.iterrows():
        hist = inv.loc[
            (inv["weekofyear"] == row["weekofyear"])
            & (inv["year"] < row["year"])
            & (inv["year"] >= row["year"] - 5)
        ]
        if len(hist) < 3 or hist["stocks"].std(ddof=0) == 0:
            zscores.append(np.nan)
        else:
            zscores.append((row["stocks"] - hist["stocks"].mean()) / hist["stocks"].std(ddof=0))
    inv["inv_seasonal_z"] = zscores

    left = frame[["retail_monday", "prediction_ts_utc"]].sort_values("prediction_ts_utc")
    right = inv[
        [
            "inventory_release_ts_utc",
            "inventory_feature_timestamp",
            "inv_wow",
            "inv_seasonal_z",
        ]
    ].sort_values("inventory_release_ts_utc")
    inv_join = pd.merge_asof(
        left,
        right,
        left_on="prediction_ts_utc",
        right_on="inventory_release_ts_utc",
        direction="backward",
    )
    frame = frame.merge(
        inv_join[
            [
                "retail_monday",
                "inventory_feature_timestamp",
                "inventory_release_ts_utc",
                "inv_wow",
                "inv_seasonal_z",
            ]
        ],
        on="retail_monday",
        how="left",
    )

    doy = frame["prediction_date"].dt.dayofyear
    frame["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    frame["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    frame["is_summer"] = frame["prediction_date"].map(_is_summer).astype(int)
    frame["is_holiday_travel_week"] = frame["prediction_date"].map(_holiday_travel_week).astype(int)
    frame["month"] = frame["prediction_date"].dt.month

    required = ALL_MODEL_FEATURES + MISMATCH_FEATURES + ["target", "spot_feature_timestamp", "inventory_feature_timestamp"]
    missing_mask = frame[required].isna().any(axis=1)
    for _, row in frame.loc[missing_mask].iterrows():
        missing_cols = [c for c in required if pd.isna(row[c])]
        drop_records.append(
            {
                "retail_monday": row["retail_monday"],
                "reason": "missing_required_features:" + ",".join(missing_cols[:8]),
            }
        )
    kept = frame.loc[~missing_mask].copy()
    _assert_no_leakage(kept)
    keep_cols = TIMESTAMP_COLS + ALL_MODEL_FEATURES + MISMATCH_FEATURES + DIAGNOSTIC_HUB_FEATURES + ["retail_last"]
    kept = kept[keep_cols].sort_values("prediction_date").reset_index(drop=True)
    dropped = pd.DataFrame(drop_records)
    logger.info("%s kept=%s dropped=%s", region.region_id, len(kept), len(dropped))
    return DatasetBuildResult(frame=kept, dropped=dropped)
