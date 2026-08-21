"""Feature-group configuration for Gas Wait modeling experiments.

Experiments select groups by name without rewriting feature code::

    feature_groups = ["RETAIL_MOMENTUM", "CRUDE", "WHOLESALE"]
    columns = resolve_feature_columns(feature_groups)
"""

from __future__ import annotations

from enum import Enum


class FeatureGroup(str, Enum):
    RETAIL_MOMENTUM = "RETAIL_MOMENTUM"
    CRUDE = "CRUDE"
    WHOLESALE = "WHOLESALE"
    SPREADS = "SPREADS"
    INVENTORY = "INVENTORY"
    SEASONALITY = "SEASONALITY"


# Columns produced for the weekly EIA experiment (exp01) when all groups are enabled.
# Additional framework columns (e.g. wti_d7, retail_chg_p4) exist but are not in exp01.
WEEKLY_EIA_FEATURE_COLUMNS: dict[FeatureGroup, tuple[str, ...]] = {
    FeatureGroup.RETAIL_MOMENTUM: ("retail_d7", "retail_d14"),
    FeatureGroup.CRUDE: ("wti_d1", "wti_d3", "wti_d5", "wti_vol_20"),
    FeatureGroup.WHOLESALE: ("nyh_d1", "nyh_d5", "nyh_vol_20", "gc_d1", "gc_d5"),
    FeatureGroup.SPREADS: ("crack_nyh_d5",),
    FeatureGroup.INVENTORY: ("inv_wow", "inv_seasonal_z"),
    FeatureGroup.SEASONALITY: ("sin_doy", "cos_doy", "is_summer"),
}

# Extended columns the framework can compute (future daily retail experiments).
EXTENDED_FEATURE_COLUMNS: dict[FeatureGroup, tuple[str, ...]] = {
    FeatureGroup.RETAIL_MOMENTUM: (
        "retail_d7",
        "retail_d14",
        "retail_chg_p4",
    ),
    FeatureGroup.CRUDE: (
        "wti_d1",
        "wti_d3",
        "wti_d5",
        "wti_d7",
        "wti_vol_20",
        "wti_ma_dist_20",
        "wti_mom_5_20",
    ),
    FeatureGroup.WHOLESALE: (
        "nyh_d1",
        "nyh_d3",
        "nyh_d5",
        "nyh_d7",
        "nyh_vol_20",
        "nyh_ma_dist_20",
        "gc_d1",
        "gc_d3",
        "gc_d5",
        "gc_d7",
        "gc_vol_20",
        "gc_ma_dist_20",
    ),
    FeatureGroup.SPREADS: (
        "crack_nyh_d5",
        "spread_nyh_gc",
        "spread_nyh_gc_d5",
    ),
    FeatureGroup.INVENTORY: ("inv_wow", "inv_yoy", "inv_seasonal_z"),
    FeatureGroup.SEASONALITY: ("month", "weekofyear", "sin_doy", "cos_doy", "is_summer"),
}

DEFAULT_FEATURE_GROUPS: tuple[FeatureGroup, ...] = tuple(FeatureGroup)

# Research dead-zone thresholds (not tuned on test).
DEFAULT_SIGNAL_THRESHOLDS: tuple[float, ...] = (0.03, 0.04, 0.05)
DEFAULT_SIGNAL_THRESHOLD = 0.03

# Theoretical fill size for economics (not a behavioral claim).
DEFAULT_TANK_GALLONS = 15.0


def parse_feature_groups(names: list[str] | tuple[str, ...]) -> list[FeatureGroup]:
    """Parse string group names into ``FeatureGroup`` values."""
    out: list[FeatureGroup] = []
    for name in names:
        out.append(FeatureGroup(name.upper()))
    return out


def resolve_feature_columns(
    groups: list[FeatureGroup] | list[str],
    *,
    extended: bool = False,
) -> list[str]:
    """Return ordered feature column names for the selected groups."""
    if groups and isinstance(groups[0], str):
        groups = parse_feature_groups(groups)  # type: ignore[assignment]
    catalog = EXTENDED_FEATURE_COLUMNS if extended else WEEKLY_EIA_FEATURE_COLUMNS
    cols: list[str] = []
    for group in groups:
        for col in catalog[group]:
            if col not in cols:
                cols.append(col)
    return cols
