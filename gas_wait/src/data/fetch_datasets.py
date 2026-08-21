"""Download verified EIA datasets and write raw/processed files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .datasets import ALL_DATASETS, EIADatasetDefinition, REGIONAL_RETAIL_DATASETS
from .eia_client import EIAClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def clean_dataset(
    df: pd.DataFrame,
    definition: EIADatasetDefinition,
) -> pd.DataFrame:
    """Add standardized metadata columns without changing the native frequency."""
    if df.empty:
        return df

    cleaned = df.copy()
    cleaned["dataset_id"] = definition.dataset_id
    cleaned["dataset_name"] = definition.name
    cleaned["source"] = definition.source
    cleaned["frequency"] = definition.frequency
    cleaned["geographic_area"] = definition.geographic_area
    cleaned["units"] = definition.units
    cleaned["category"] = definition.category

    if definition.legacy_series_id:
        cleaned["legacy_series_id"] = definition.legacy_series_id

    column_order = [
        "observation_date",
        "period",
        "value",
        "dataset_id",
        "dataset_name",
        "source",
        "frequency",
        "geographic_area",
        "units",
        "category",
        "legacy_series_id",
    ]
    existing = [column for column in column_order if column in cleaned.columns]
    remaining = [column for column in cleaned.columns if column not in existing]
    return cleaned[existing + remaining]


def fetch_dataset(
    client: EIAClient,
    definition: EIADatasetDefinition,
    *,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, object]:
    """Fetch one dataset and persist raw JSON plus cleaned CSV/metadata."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    payload = client.fetch_all_data(
        definition.route,
        frequency=definition.frequency,
        data_columns=definition.data_columns,
        facets=definition.facets,
        sort_column="period",
        sort_direction="asc",
    )

    raw_path = raw_dir / f"{definition.dataset_id}.json"
    processed_path = processed_dir / f"{definition.dataset_id}.csv"
    metadata_path = processed_dir / f"{definition.dataset_id}.metadata.json"

    client.save_raw_json(payload, str(raw_path))

    df = client.response_to_dataframe(payload)
    cleaned = clean_dataset(df, definition)
    cleaned.to_csv(processed_path, index=False)

    metadata = {
        **definition.to_metadata(),
        "eia_series_facet": _series_facet(definition),
        "observation_count": int(len(cleaned)),
        "date_range": _date_range_summary(cleaned),
        "missing_values": _missing_value_summary(cleaned),
        "frequency_check": _frequency_check(cleaned, definition.frequency),
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "processed_path": str(processed_path.relative_to(PROJECT_ROOT)),
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Saved %s (%s rows)", definition.dataset_id, len(cleaned))
    return metadata


def fetch_regional_retail_datasets(client: EIAClient | None = None) -> list[dict[str, object]]:
    """Fetch EIA weekly regional/city regular retail series used in exp02.

    Does not re-fetch or overwrite the original national weekly or daily spot files
    unless those dataset_ids are passed in separately.
    """
    client = client or EIAClient()
    results: list[dict[str, object]] = []
    for definition in REGIONAL_RETAIL_DATASETS:
        logger.info("Fetching regional retail: %s", definition.dataset_id)
        results.append(fetch_dataset(client, definition))
    return results


def fetch_all_datasets(client: EIAClient | None = None) -> list[dict[str, object]]:
    """Fetch all active EIA datasets (weekly and daily)."""
    client = client or EIAClient()
    results: list[dict[str, object]] = []

    for definition in ALL_DATASETS:
        logger.info("Fetching dataset: %s", definition.dataset_id)
        results.append(fetch_dataset(client, definition))

    return results


def _series_facet(definition: EIADatasetDefinition) -> str | None:
    series_values = definition.facets.get("series")
    if series_values:
        return series_values[0]
    return definition.legacy_series_id


def _frequency_check(df: pd.DataFrame, expected_frequency: str) -> dict[str, object]:
    if df.empty or "observation_date" not in df.columns:
        return {
            "expected_frequency": expected_frequency,
            "median_gap_days": None,
            "pct_gap_le_3_days": None,
            "is_native_frequency": False,
        }

    gaps = df["observation_date"].sort_values().diff().dt.days.dropna()
    if gaps.empty:
        return {
            "expected_frequency": expected_frequency,
            "median_gap_days": None,
            "pct_gap_le_3_days": None,
            "is_native_frequency": False,
        }

    median_gap = float(gaps.median())
    pct_short_gap = float((gaps <= 3).mean())

    if expected_frequency == "daily":
        is_native = median_gap <= 3 and pct_short_gap >= 0.95
    elif expected_frequency == "weekly":
        is_native = median_gap == 7 and float((gaps == 7).mean()) >= 0.99
    else:
        is_native = True

    return {
        "expected_frequency": expected_frequency,
        "median_gap_days": median_gap,
        "pct_gap_le_3_days": pct_short_gap,
        "is_native_frequency": is_native,
    }


def _date_range_summary(df: pd.DataFrame) -> dict[str, str | None]:
    if df.empty or "observation_date" not in df.columns:
        return {"start": None, "end": None}

    valid_dates = df["observation_date"].dropna()
    if valid_dates.empty:
        return {"start": None, "end": None}

    return {
        "start": valid_dates.min().date().isoformat(),
        "end": valid_dates.max().date().isoformat(),
    }


def _missing_value_summary(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"value": 0, "observation_date": 0}

    summary: dict[str, int] = {}
    for column in ("value", "observation_date"):
        if column in df.columns:
            summary[column] = int(df[column].isna().sum())
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fetch_all_datasets()
