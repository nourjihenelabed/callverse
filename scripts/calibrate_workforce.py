"""Fit workforce calibration values from the Technion Anonymous Bank files.

Usage:
    python scripts/calibrate_workforce.py                 # fit everything from the raw files
    python scripts/calibrate_workforce.py --roster-only   # only resize the roster from calibration.json

The raw files are the twelve tab-separated monthly files from the Technion
Anonymous Bank archive. They are intentionally kept out of Git.

Two facts about the data matter here (see the documentation PDF):
- dates are written YYMMDD with a two-digit year: 990901 is 1 September 1999;
- the bank is fully staffed Sunday to Thursday; it closes at 14:00 on Friday and
  only reopens around 20:00 on Saturday. The arrival curve uses Sunday-Thursday only.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = ROOT / "data" / "raw" / "technion"
DEFAULT_OUTPUT = ROOT / "workforce" / "calibration.json"
QUEUE_TYPES = {
    "general": {"PS", "PE"},
    "technical": {"IN"},
    "complaint": {"NE"},
}
ALL_TYPES = set().union(*QUEUE_TYPES.values())
QUEUES = ("general", "technical", "complaint")
WORKING_DAYS = {6, 0, 1, 2, 3}         # Python weekday(): Sunday=6, Monday=0 ... Thursday=3
ROSTER_PEAK_USE = 0.8                  # the Erlang C plan uses at most this share of the roster at peak


def _files(input_dir: Path, expected: int | None = 12) -> list[Path]:
    files = sorted(input_dir.glob("*1999.txt"))
    if expected is not None and len(files) != expected:
        raise SystemExit(f"Expected {expected} monthly Technion files in {input_dir}, found {len(files)}")
    if not files:
        raise SystemExit(f"No Technion files (*1999.txt) in {input_dir}")
    return files


def _queue(call_type: str) -> str | None:
    return next((queue for queue, types in QUEUE_TYPES.items() if call_type in types), None)


def _parse_date(value: str) -> date:
    """YYMMDD with a two-digit year from the 1990s: 990901 -> 1999-09-01."""
    value = value.strip()
    return date(1900 + int(value[:2]), int(value[2:4]), int(value[4:6]))


def _working_day(day: date) -> bool:
    """The bank's fully staffed days: Sunday to Thursday."""
    return day.weekday() in WORKING_DAYS


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


def size_roster(calls_per_hour, mean_handle_minutes: dict, call_mix: dict | None = None,
                sla_seconds: float | None = None, sla_target: float | None = None,
                peak_use: float = ROSTER_PEAK_USE) -> dict:
    """Roster such that the Erlang C plan uses at most `peak_use` of it at its busiest hour,
    for each skill: enough technical-skilled advisors for the technical peak, enough
    complaint-skilled ones for the complaint peak, and enough people in total."""
    from workforce.manager import advisors_needed
    from workforce.simulation import CenterConfig

    defaults = CenterConfig.__dataclass_fields__
    call_mix = call_mix or defaults["call_mix"].default_factory()
    sla_seconds = sla_seconds or defaults["sla_seconds"].default
    sla_target = sla_target or defaults["sla_target"].default
    plan = [
        {q: advisors_needed(calls * call_mix[q], mean_handle_minutes[q], sla_seconds, sla_target) for q in QUEUES}
        for calls in calls_per_hour
    ]
    peak_total = max(sum(hour.values()) for hour in plan)
    peak_technical = max(hour["technical"] for hour in plan)
    peak_complaint = max(hour["complaint"] for hour in plan)
    technical = math.ceil(peak_technical / peak_use)
    complaint = math.ceil(peak_complaint / peak_use)
    total = max(math.ceil(peak_total / peak_use), technical + complaint)
    return {
        "general_only": total - technical - complaint,
        "general_technical": technical,
        "general_complaint": complaint,
        "erlang_peak": {"total": peak_total, "technical": peak_technical, "complaint": peak_complaint},
        "source": (f"Sized so the Erlang C plan uses at most {peak_use:.0%} of the roster at its busiest hour, "
                   "per skill (calls_per_hour, call_mix and service times of this file)."),
    }


def calibrate(input_dir: Path = DEFAULT_INPUT, expected_files: int | None = 12) -> dict:
    hourly_arrivals = Counter()
    working_days: defaultdict[date, int] = defaultdict(int)
    service_times: defaultdict[str, list[float]] = defaultdict(list)
    patience_exposure: defaultdict[int, float] = defaultdict(float)
    patience_abandonments: Counter[int] = Counter()
    records = 0

    files = _files(input_dir, expected_files)
    for path in files:
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
                if q_start != "0:00:00" and _working_day(day):
                    hourly_arrivals[_hour(q_start)] += 1
                    working_days[day] += 1
                q_time = float(row["q_time"])
                if row["outcome"] == "AGENT":
                    service_time = float(row["ser_time"])
                    if service_time >= 10:
                        service_times[queue].append(service_time)
                    if q_start != "0:00:00":
                        patience_exposure[_hour(q_start)] += q_time
                elif row["outcome"] == "HANG" and q_start != "0:00:00":
                    patience_exposure[_hour(q_start)] += q_time
                    patience_abandonments[_hour(q_start)] += 1

    days = sorted(working_days)
    if not days:
        raise SystemExit("No Sunday-Thursday queue arrivals found")
    day_count = len(days)
    arrivals = [hourly_arrivals[hour] / day_count for hour in range(24)]
    open_hour, close_hour = 8, 20
    raw_calls_per_hour = arrivals[open_hour:close_hour]
    target_daily_calls = 1050.0
    scale = target_daily_calls / sum(raw_calls_per_hour)
    calls_per_hour = [value * scale for value in raw_calls_per_hour]
    daily_volumes = list(working_days.values())
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
    total_hang_ups = sum(patience_abandonments.values())
    if not total_hang_ups:
        raise SystemExit("No HANG calls found: cannot estimate patience")
    # overall mean = all waiting time / all hang-ups (not an average of the hourly values,
    # which would give night hours with a handful of calls the same weight as busy hours)
    overall_patience = sum(patience_exposure.values()) / total_hang_ups / 60

    return {
        "phase": 2,
        "source": {
            "dataset": "Technion Anonymous Bank",
            "files": len(files),
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
            "working_day_count": day_count,
            "daily_volume_cv": round(daily_spread, 6),
            "source": "Queue entries on the bank's working days (Sunday-Thursday), grouped by hour, 1999 monthly files.",
        },
        "service": service,
        "patience": {
            "minutes": {queue: round(overall_patience, 6) for queue in QUEUES},
            "by_hour": patience,
            "source": "Censored exponential estimate: total queue-time exposure divided by HANG events, by queue-entry hour.",
        },
        "forecast_error": {
            "maximum": 0.2,
            "source": "Retained from Phase 1 Brown et al. published forecast variability.",
        },
        "roster": size_roster(calls_per_hour, {q: service[q]["mean_minutes"] for q in QUEUES}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit workforce values from Technion Anonymous Bank files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roster-only", action="store_true",
                        help="keep calibration.json as it is and only resize its roster (no raw files needed)")
    args = parser.parse_args()
    if args.roster_only:
        result = json.loads(args.output.read_text(encoding="utf-8"))
        result["roster"] = size_roster(result["arrivals"]["calls_per_hour"],
                                       {q: result["service"][q]["mean_minutes"] for q in QUEUES})
    else:
        result = calibrate(args.input_dir)
        print(f"Read {result['source']['records_read']:,} records from {result['source']['files']} files, "
              f"{result['arrivals']['working_day_count']} working days (Sunday-Thursday).")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    r = result["roster"]
    print(f"Roster: {r['general_only']} general only, {r['general_technical']} general + technical, "
          f"{r['general_complaint']} general + complaint "
          f"(Erlang C peak: {r['erlang_peak']['total']} of {r['general_only'] + r['general_technical'] + r['general_complaint']})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
