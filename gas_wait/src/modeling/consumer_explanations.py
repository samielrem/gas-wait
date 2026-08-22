"""Map model explanation bullets to consumer-facing wording.

Does not change model math, coefficients, or which groups are selected.
"""

from __future__ import annotations

CONSUMER_EXPLANATION_MAP: dict[str, str] = {
    "Recent retail momentum is negative": "Recent pump prices have been moving down.",
    "Recent retail momentum is positive": "Recent pump prices have been moving up.",
    "WTI is falling": "Crude oil prices have been falling.",
    "WTI is rising": "Crude oil prices have been rising.",
    "Wholesale gasoline is falling": "Gasoline prices are falling in the wholesale market.",
    "Wholesale gasoline is rising": "Gasoline prices are rising in the wholesale market.",
    "Crack spread is narrowing": "Gasoline is becoming cheaper before it reaches the pump.",
    "Crack spread is widening": "Gasoline is becoming more expensive before it reaches the pump.",
    "Gasoline inventories rose week-over-week": "Gasoline supplies rose last week.",
    "Gasoline inventories fell week-over-week": "Gasoline supplies fell last week.",
    "Retail momentum inputs lean toward higher prices": "Recent pump prices lean toward higher prices.",
    "Retail momentum inputs lean toward lower prices": "Recent pump prices lean toward lower prices.",
    "WTI inputs lean toward higher prices": "Crude oil markets lean toward higher prices.",
    "WTI inputs lean toward lower prices": "Crude oil markets lean toward lower prices.",
    "Wholesale gasoline inputs lean toward higher prices": "Wholesale gasoline markets lean toward higher prices.",
    "Wholesale gasoline inputs lean toward lower prices": "Wholesale gasoline markets lean toward lower prices.",
    "Crack spread inputs lean toward higher prices": "Gasoline is becoming more expensive before it reaches the pump.",
    "Crack spread inputs lean toward lower prices": "Gasoline is becoming cheaper before it reaches the pump.",
    "Inventory inputs lean toward higher prices": "Supply conditions lean toward higher prices.",
    "Inventory inputs lean toward lower prices": "Supply conditions lean toward lower prices.",
    "Seasonality inputs lean toward higher prices": "Typical seasonal patterns lean toward higher prices.",
    "Seasonality inputs lean toward lower prices": "Typical seasonal patterns lean toward lower prices.",
}


def to_consumer_explanations(explanations: tuple[str, ...] | list[str]) -> list[str]:
    """Return consumer-safe bullets; drop any leftover technical phrasing."""
    out: list[str] = []
    for item in explanations:
        mapped = CONSUMER_EXPLANATION_MAP.get(item)
        if mapped:
            out.append(mapped)
    return out
