"""Fit workforce calibration values from the Technion Anonymous Bank files.

Usage:
    python scripts/calibrate_workforce.py

The raw files are the twelve tab-separated monthly files from the Technion
Anonymous Bank archive. They are intentionally kept out of Git.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "technion"
DEFAULT_OUTPUT = ROOT / "workforce" / "calibration.json"
QUEUE_TYPES = {
    "general": {"PS", "PE"},
    "technical": {"IN"},
    "complaint": {"NE"},
}
ALL_TYPES = set().union(*QUEUE_TYPES.values())


def _files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*1999.txt"))
    if len(files) != 12:
        raise SystemExit(f"Expected 12 monthly Technion files in {input_dir}, found {len(files)}")
    return files


def _queue(call_type: str) -> str | None:
    return next((queue for queue, types in QUEUE_TYPES.items() if call_type in types), None)


def _parse_date(value: str) -> date:
    return date(2000 + int(value[:2]), int(value[2:4]), int(value[4:6]))


def _hour(value: str) -> int:
    return int(value.split(":", 1)[0])


def _mean_and_spread(seconds: list[float]) -> tuple[float, float]:
    if not seconds:
        raise ValueError("Cannot fit service duration without observations")
    logs = [math.log(value) for value in seconds if value >= 10]
    if not logs:
        raise ValueError("Cannot fit service duration without calls of at least 10 seconds")
    mean_log = sum(logs) / len(logs)
    variance = sum((value - mean_log) ** 2 for value in logs) / len(logs)
    filtered_seconds = [value for value in seconds if value >= 10]
    mean_seconds = sum(filtered_seconds) / len(filtered_seconds)
    return mean_seconds / 60, math.sqrt(variance)


def calibrate(input_dir: Path = DEFAULT_INPUT) -> dict:
    hourly_arrivals = Counter()
    weekday_days: defaultdict[date, int] = defaultdict(int)
    service_times: defaultdict[str, list[float]] = defaultdict(list)
    patience_exposure: defaultdict[int, float] = defaultdict(float)
    patience_abandonments: Counter[int] = Counter()
    records = 0

    for path in _files(input_dir):
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                records += 1
                call_type = row["type"]
                queue = _queue(call_type)
                if queue is None or row["outcome"] == "PHANTOM":
                    continue
                day = _parse_date(row["date"])
                q_start = row["q_start"]
                if q_start != "0:00:00":
                    if day.weekday() < 5:
                        hourly_arrivals[_hour(q_start)] += 1
                        weekday_days[day] += 1
                q_time = float(row["q_time"])
                if row["outcome"] == "AGENT":
                    service_time = float(row["ser_time"])
                    if service_time >= 10:
                        service_times[queue].append(service_time)
                    patience_exposure[_hour(q_start)] += q_time
                elif row["outcome"] == "HANG" and q_start != "0:00:00":
                    patience_exposure[_hour(q_start)] += q_time
                    patience_abandonments[_hour(q_start)] += 1

    weekdays = sorted(weekday_days)
    if not weekdays:
        raise SystemExit("No weekday queue arrivals found")
    day_count = len(weekdays)
    arrivals = [hourly_arrivals[hour] / day_count for hour in range(24)]
    open_hour, close_hour = 8, 20
    raw_calls_per_hour = arrivals[open_hour:close_hour]
    target_daily_calls = 1050.0
    scale = target_daily_calls / sum(raw_calls_per_hour)
    calls_per_hour = [value * scale for value in raw_calls_per_hour]
    daily_volumes = list(weekday_days.values())
    daily_mean = sum(daily_volumes) / len(daily_volumes)
    daily_spread = (
        math.sqrt(sum((value - daily_mean) ** 2 for value in daily_volumes) / len(daily_volumes))
        / daily_mean
        if daily_mean else 0.0
    )

    service = {}
    for queue in ("general", "technical", "complaint"):
        if queue == "complaint":
            longest_type = max(service_times, key=lambda value: sum(service_times[value]) / len(service_times[value]))
            mean_minutes, spread = _mean_and_spread(service_times[longest_type])
            service[queue] = {
                "mean_minutes": round(mean_minutes, 6),
                "spread": round(spread, 6),
                "observations": len(service_times[longest_type]),
            }
            service[queue]["source"] = (
                f"Temporary complaint proxy: longest observed type, {longest_type}, "
                "from the Technion Anonymous Bank files."
            )
            continue
        mean_minutes, spread = _mean_and_spread(service_times[queue])
        service[queue] = {
            "mean_minutes": round(mean_minutes, 6),
            "spread": round(spread, 6),
            "observations": len(service_times[queue]),
            "source": f"Technion Anonymous Bank raw files; AGENT calls mapped to {queue}",
        }

    patience = {}
    for hour in range(24):
        events = patience_abandonments[hour]
        patience[hour] = round(patience_exposure[hour] / events / 60, 6) if events else None
    observed_patience = [value for value in patience.values() if value is not None]
    fallback_patience = sum(observed_patience) / len(observed_patience)

    return {
        "phase": 2,
        "source": {
            "dataset": "Technion Anonymous Bank",
            "files": 12,
            "records_read": records,
            "input_dir": "data/raw/technion",
            "documentation": "https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.pdf",
        },
        "arrivals": {
            "open_hour": open_hour,
            "close_hour": close_hour,
            "calls_per_hour": [round(value, 6) for value in calls_per_hour],
            "raw_calls_per_hour": [round(value, 6) for value in raw_calls_per_hour],
            "rescaled_to_daily_calls": target_daily_calls,
            "weekday_count": day_count,
            "daily_volume_cv": round(daily_spread, 6),
            "source": "Weekday queue entries grouped by hour from the 1999 monthly files.",
        },
        "service": service,
        "patience": {
            "minutes": {queue: round(fallback_patience, 6) for queue in ("general", "technical", "complaint")},
            "by_hour": patience,
            "source": "Censored exponential estimate: total queue-time exposure divided by HANG events, by queue-entry hour.",
        },
        "forecast_error": {
            "maximum": 0.2,
            "source": "Retained from Phase 1 Brown et al. published forecast variability.",
        },
        "roster": {
            "general_only": 6,
            "general_technical": 9,
            "general_complaint": 6,
            "source": "Retained from Phase 1 roster sizing rule.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit workforce values from Technion Anonymous Bank files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = calibrate(args.input_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Read {result['source']['records_read']:,} records from {result['source']['files']} files.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
