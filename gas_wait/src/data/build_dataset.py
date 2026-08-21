"""Load downloaded EIA datasets and report pipeline status."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .datasets import ALL_DATASETS, ACTIVE_DAILY_DATASETS, ACTIVE_WEEKLY_DATASETS
from .eia_client import EIAClient, EIAMissingAPIKeyError
from .fetch_datasets import fetch_all_datasets

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_processed_dataset(dataset_id: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = processed_dir / f"{dataset_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["observation_date"])
    return df


def summarize_dataset(dataset_id: str, processed_dir: Path = PROCESSED_DIR) -> dict[str, object]:
    """Summarize one processed dataset, including date range and missing values."""
    metadata_path = processed_dir / f"{dataset_id}.metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as handle:
            return json.load(handle)

    df = load_processed_dataset(dataset_id, processed_dir)
    valid_dates = df["observation_date"].dropna() if "observation_date" in df.columns else pd.Series(dtype="datetime64[ns]")

    return {
        "dataset_id": dataset_id,
        "observation_count": int(len(df)),
        "frequency": df["frequency"].iloc[0] if "frequency" in df.columns and not df.empty else None,
        "date_range": {
            "start": valid_dates.min().date().isoformat() if not valid_dates.empty else None,
            "end": valid_dates.max().date().isoformat() if not valid_dates.empty else None,
        },
        "missing_values": {
            "value": int(df["value"].isna().sum()) if "value" in df.columns else 0,
            "observation_date": int(df["observation_date"].isna().sum()) if "observation_date" in df.columns else 0,
        },
    }


def report_all_datasets(processed_dir: Path = PROCESSED_DIR) -> list[dict[str, object]]:
    """Load all datasets and report their ranges and frequencies."""
    summaries: list[dict[str, object]] = []

    for definition in ALL_DATASETS:
        try:
            summary = summarize_dataset(definition.dataset_id, processed_dir)
            summaries.append(summary)
        except FileNotFoundError:
            summaries.append(
                {
                    **definition.to_metadata(),
                    "eia_series_facet": definition.facets.get("series", [None])[0]
                    if definition.facets.get("series")
                    else definition.legacy_series_id,
                    "observation_count": None,
                    "date_range": {"start": None, "end": None},
                    "missing_values": {},
                    "status": "missing",
                }
            )

    return summaries


def report_dataset_groups(processed_dir: Path = PROCESSED_DIR) -> dict[str, list[dict[str, object]]]:
    """Report weekly and daily datasets separately."""
    all_summaries = report_all_datasets(processed_dir)
    by_id = {summary["dataset_id"]: summary for summary in all_summaries if summary.get("dataset_id")}

    return {
        "weekly": [by_id[definition.dataset_id] for definition in ACTIVE_WEEKLY_DATASETS if definition.dataset_id in by_id],
        "daily": [by_id[definition.dataset_id] for definition in ACTIVE_DAILY_DATASETS if definition.dataset_id in by_id],
    }


def print_report(summaries: list[dict[str, object]], *, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
        print("-" * 72)
    else:
        print("\nGas Wait EIA dataset report")
        print("=" * 72)

    for summary in summaries:
        dataset_id = summary.get("dataset_id")
        print(f"\nDataset: {dataset_id}")
        print(f"  Name:       {summary.get('name', 'n/a')}")
        print(f"  Frequency:  {summary.get('frequency', 'n/a')}")
        print(f"  EIA series: {summary.get('eia_series_facet', summary.get('legacy_series_id', 'n/a'))}")
        print(f"  Source:     {summary.get('source', 'n/a')}")
        print(f"  Area:       {summary.get('geographic_area', 'n/a')}")
        print(f"  Units:      {summary.get('units', 'n/a')}")
        print(f"  Obs count:  {summary.get('observation_count', 'n/a')}")

        date_range = summary.get("date_range", {})
        if isinstance(date_range, dict):
            print(f"  Date range: {date_range.get('start')} to {date_range.get('end')}")

        missing_values = summary.get("missing_values", {})
        if isinstance(missing_values, dict) and missing_values:
            print(f"  Missing:    {missing_values}")

        frequency_check = summary.get("frequency_check", {})
        if isinstance(frequency_check, dict) and frequency_check:
            print(
                "  Freq check: "
                f"median_gap={frequency_check.get('median_gap_days')} days, "
                f"pct_gap<=3={frequency_check.get('pct_gap_le_3_days')}, "
                f"native={frequency_check.get('is_native_frequency')}"
            )

        if summary.get("status") == "missing":
            print("  Status:     NOT DOWNLOADED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and report Gas Wait EIA datasets.")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch datasets from the EIA API before reporting.",
    )
    parser.add_argument(
        "--daily-only",
        action="store_true",
        help="Fetch only the daily petroleum spot datasets.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.fetch:
        try:
            if args.daily_only:
                from .fetch_datasets import fetch_dataset

                client = EIAClient()
                for definition in ACTIVE_DAILY_DATASETS:
                    logger.info("Fetching dataset: %s", definition.dataset_id)
                    fetch_dataset(client, definition)
            else:
                fetch_all_datasets()
        except EIAMissingAPIKeyError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc

    groups = report_dataset_groups()
    print_report(groups["weekly"], title="Weekly datasets")
    print_report(groups["daily"], title="Daily petroleum spot datasets")


if __name__ == "__main__":
    main()
