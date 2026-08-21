"""Discover EIA weekly regular retail gasoline geographies (coverage catalog).

Uses the existing EIA API only. Does not change first-model source datasets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _load_env() -> None:
    for path in (WORKSPACE_ROOT / ".env", PROJECT_ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def discover_weekly_regular_retail() -> pd.DataFrame:
    from data.eia_client import EIAClient

    _load_env()
    client = EIAClient()
    facet_payload = client._request("petroleum/pri/gnd/facet/duoarea")
    facet_rows = facet_payload.get("response", {}).get("facets", [])
    facet_df = pd.DataFrame(facet_rows)
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    facet_path = out_dir / "eia_retail_duoarea_facets.csv"
    facet_df.to_csv(facet_path, index=False)

    payload = client.fetch_all_data(
        "petroleum/pri/gnd/data",
        frequency="weekly",
        data_columns=["value"],
        facets={"product": ["EPMR"], "process": ["PTE"]},
        sort_column="period",
        sort_direction="asc",
    )
    df = client.response_to_dataframe(payload)
    if df.empty:
        raise RuntimeError("EIA returned no weekly regular retail rows.")

    coverage = (
        df.dropna(subset=["observation_date"])
        .groupby(["duoarea", "area-name"], dropna=False)
        .agg(
            n=("value", "size"),
            n_missing=("value", lambda s: int(s.isna().sum())),
            start=("observation_date", "min"),
            end=("observation_date", "max"),
        )
        .reset_index()
        .sort_values(["start", "duoarea"])
    )
    coverage["start"] = coverage["start"].dt.date.astype(str)
    coverage["end"] = coverage["end"].dt.date.astype(str)
    coverage_path = out_dir / "eia_weekly_regular_retail_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    print(f"Wrote {facet_path}")
    print(f"Wrote {coverage_path} ({len(coverage)} geographies)")
    print(coverage.to_string(index=False))
    return coverage


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    discover_weekly_regular_retail()
