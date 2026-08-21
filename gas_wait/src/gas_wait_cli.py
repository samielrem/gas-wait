"""Personal MVP command-line tool for weekly Gas Wait signals."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from modeling.inference import (
    MODEL_ID,
    MODEL_NAME,
    MODEL_VERSION,
    SignalForecast,
    generate_weekly_signal,
)
from modeling.weekly_pipeline import PROCESSED_DIR

EASTERN = ZoneInfo("America/New_York")
HISTORY_PATH = PROCESSED_DIR / "signal_history.csv"
HISTORY_COLUMNS = [
    "timestamp",
    "prediction_date",
    "predicted_change",
    "signal",
    "model_version",
    "latest_data_date",
]


def _format_cents_inline(change: float) -> str:
    cents = change * 100.0
    sign = "+" if cents > 0 else ""
    return f"{sign}{cents:.1f}¢ / gallon"


def forecast_to_json(forecast: SignalForecast) -> dict[str, object]:
    """Machine-readable signal payload."""
    payload: dict[str, object] = {
        "signal": forecast.signal.signal,
        "predicted_change_dollars_per_gallon": round(forecast.predicted_change, 6),
        "predicted_change_cents_per_gallon": round(forecast.predicted_change * 100.0, 1),
        "horizon": "7 days",
        "latest_data_date": forecast.latest_data_date.strftime("%Y-%m-%d"),
        "model": MODEL_ID,
        "prediction_date": forecast.prediction_date.strftime("%Y-%m-%d"),
        "national_weekly_disclaimer": "National weekly signal — not local station pricing.",
    }
    if forecast.data_warning:
        payload["data_warning"] = forecast.data_warning
    return payload


def format_signal_output(forecast: SignalForecast) -> str:
    """Render the terminal-friendly signal block."""
    sig = forecast.signal.signal
    lines = [
        "GAS WAIT",
        "────────────────────────",
        "",
        f"Signal: {sig}",
        "",
        f"Expected next-week move: {_format_cents_inline(forecast.predicted_change)}",
        "",
        "Horizon: ~7 days",
        "",
        f"Last available data: {forecast.latest_data_date.strftime('%Y-%m-%d')}",
        "",
        "National weekly signal — not local station pricing.",
    ]
    if forecast.data_warning:
        lines.extend(["", forecast.data_warning])
    if forecast.explanations:
        lines.extend(["", "Why:"])
        for item in forecast.explanations:
            lines.append(f"• {item}")
    lines.extend(["", f"Model: {MODEL_NAME}", "", "────────────────────────"])
    return "\n".join(lines)


def append_signal_history(forecast: SignalForecast, path: Path = HISTORY_PATH) -> None:
    """Append one history row; one entry per ``prediction_date``; preserve prior rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_date = forecast.prediction_date.strftime("%Y-%m-%d")

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["prediction_date"])
        existing = existing[HISTORY_COLUMNS] if all(c in existing.columns for c in HISTORY_COLUMNS) else existing
    else:
        existing = pd.DataFrame(columns=HISTORY_COLUMNS)

    if not existing.empty:
        existing["prediction_date"] = pd.to_datetime(existing["prediction_date"])
        if (existing["prediction_date"].dt.strftime("%Y-%m-%d") == pred_date).any():
            return
        existing = existing.drop_duplicates(subset=["prediction_date"], keep="first")

    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(EASTERN).isoformat(),
                "prediction_date": pred_date,
                "predicted_change": forecast.predicted_change,
                "signal": forecast.signal.signal,
                "model_version": MODEL_VERSION,
                "latest_data_date": forecast.latest_data_date.strftime("%Y-%m-%d"),
            }
        ]
    )
    combined = pd.concat([existing, row], ignore_index=True)
    combined["prediction_date"] = pd.to_datetime(combined["prediction_date"])
    combined = combined.sort_values("prediction_date").drop_duplicates(subset=["prediction_date"], keep="first")
    combined["prediction_date"] = combined["prediction_date"].dt.strftime("%Y-%m-%d")
    combined[HISTORY_COLUMNS].to_csv(path, index=False)


def format_history_summary(path: Path = HISTORY_PATH) -> str:
    """Compact signal history counts."""
    if not path.exists():
        return "SIGNAL HISTORY\n────────────────────────\n\nNo signal history yet. Run the CLI once to create it."

    hist = pd.read_csv(path, parse_dates=["prediction_date"])
    if hist.empty:
        return "SIGNAL HISTORY\n────────────────────────\n\nNo signal history yet. Run the CLI once to create it."

    if len(hist) < 3:
        return (
            "SIGNAL HISTORY\n────────────────────────\n\n"
            "Insufficient history for a summary (fewer than 3 recorded signals)."
        )

    counts = hist["signal"].value_counts()
    wait_n = int(counts.get("WAIT", 0))
    fill_n = int(counts.get("FILL UP", 0))
    none_n = int(counts.get("NO CLEAR SIGNAL", 0))

    lines = [
        "SIGNAL HISTORY",
        "────────────────────────",
        "",
        f"WAIT        {wait_n:>3}",
        f"FILL UP     {fill_n:>3}",
        f"NO SIGNAL   {none_n:>3}",
        "",
        f"Recorded signals: {len(hist)}",
        "",
        "Note: weekly outcomes are not scored here until each target week completes.",
        "",
        "────────────────────────",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gas Wait personal weekly gasoline signal")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show compact signal history summary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the current signal as JSON",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Path to processed EIA CSV directory",
    )
    args = parser.parse_args(argv)

    if args.history:
        print(format_history_summary())
        return 0

    try:
        forecast = generate_weekly_signal(args.processed_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("Unable to generate today's signal because required data is unavailable.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(forecast_to_json(forecast), indent=2))
    else:
        print(format_signal_output(forecast))
    append_signal_history(forecast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
