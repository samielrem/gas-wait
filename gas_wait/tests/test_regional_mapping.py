"""Regional mapping and leakage tests. No live API."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.datasets import ALL_DATASETS, REGIONAL_RETAIL_DATASETS
from modeling.build_features import _assert_no_leakage
from modeling.regional_config import SKIPPED_GEOGRAPHIES, VIABLE_REGIONS


class RegionalMappingTests(unittest.TestCase):
    def test_original_fetch_set_excludes_regional_retail(self) -> None:
        original_ids = {d.dataset_id for d in ALL_DATASETS}
        regional_ids = {d.dataset_id for d in REGIONAL_RETAIL_DATASETS}
        self.assertTrue(regional_ids.isdisjoint(original_ids))

    def test_six_viable_regions(self) -> None:
        self.assertEqual(len(VIABLE_REGIONS), 6)
        ids = [r.region_id for r in VIABLE_REGIONS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_la_uses_rbob_not_harbor(self) -> None:
        la = next(r for r in VIABLE_REGIONS if r.region_id == "la_rbob")
        self.assertIn("Y05LA", la.retail.facets["duoarea"])
        self.assertIn("EER_EPMRR_PF4_Y05LA_DPG", la.matched_hub.facets["series"])
        self.assertIn("EER_EPMRU_PF4_Y35NY_DPG", la.mismatched_hub.facets["series"])

    def test_northeast_uses_ny_harbor(self) -> None:
        for region_id in ("nyc_nyh", "padd1b_nyh"):
            region = next(r for r in VIABLE_REGIONS if r.region_id == region_id)
            self.assertIn("Y35NY", region.matched_hub.facets["series"][0])
            self.assertNotIn("Y05LA", region.matched_hub.facets["series"][0])

    def test_gulf_uses_gulf_coast_spot(self) -> None:
        for region_id in ("houston_gc", "padd3_gc"):
            region = next(r for r in VIABLE_REGIONS if r.region_id == region_id)
            self.assertIn("RGC", region.matched_hub.facets["series"][0])

    def test_padd5_and_seattle_are_skipped(self) -> None:
        skipped = {row["duoarea"] for row in SKIPPED_GEOGRAPHIES}
        self.assertIn("R50", skipped)
        self.assertIn("Y48SE", skipped)
        self.assertIn("R10", skipped)
        self.assertIn("YMIA", skipped)

    def test_leakage_on_built_file_if_present(self) -> None:
        path = SRC_DIR.parent / "data" / "processed" / "regional_weekly_model_nyc_nyh.csv"
        if not path.exists():
            self.skipTest("regional dataset not built yet")
        import pandas as pd

        df = pd.read_csv(path, parse_dates=["prediction_date", "retail_monday", "spot_feature_timestamp", "target_timestamp", "inventory_feature_timestamp", "prediction_ts_utc", "inventory_release_ts_utc"])
        _assert_no_leakage(df)


if __name__ == "__main__":
    unittest.main()
