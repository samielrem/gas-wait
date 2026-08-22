"""Consumer wording for model explanation bullets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from modeling.consumer_explanations import to_consumer_explanations


class ConsumerExplanationTests(unittest.TestCase):
    def test_maps_technical_bullets(self) -> None:
        self.assertEqual(
            to_consumer_explanations(
                (
                    "Wholesale gasoline is rising",
                    "Crack spread is widening",
                    "Gasoline inventories fell week-over-week",
                )
            ),
            [
                "Gasoline prices are rising in the wholesale market.",
                "Gasoline is becoming more expensive before it reaches the pump.",
                "Gasoline supplies fell last week.",
            ],
        )

    def test_drops_unknown_technical_text(self) -> None:
        self.assertEqual(
            to_consumer_explanations(("unknown ridge coefficient", "WTI is rising")),
            ["Crude oil prices have been rising."],
        )


if __name__ == "__main__":
    unittest.main()
