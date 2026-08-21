"""Feature group configuration tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from modeling.config import FeatureGroup, resolve_feature_columns


class ConfigTests(unittest.TestCase):
    def test_resolve_subset(self) -> None:
        cols = resolve_feature_columns(["RETAIL_MOMENTUM", "CRUDE"])
        self.assertIn("retail_d7", cols)
        self.assertIn("wti_d1", cols)
        self.assertNotIn("inv_wow", cols)

    def test_all_groups(self) -> None:
        cols = resolve_feature_columns(list(FeatureGroup))
        self.assertEqual(len(cols), 17)


if __name__ == "__main__":
    unittest.main()
