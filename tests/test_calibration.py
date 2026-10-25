"""Checks scripts/calibrate_workforce.py on small sample files in the Technion format.

No real data needed: the samples follow the documented layout (tab-separated,
one call per line, dates written YYMMDD).
"""

import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("calibrate_workforce", ROOT / "scripts" / "calibrate_workforce.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)

HEADER = ["vru+line", "call_id", "customer_id", "priority", "type", "date", "vru_entry", "vru_exit",
          "vru_time", "q_start", "q_exit", "q_time", "outcome", "ser_start", "ser_exit", "ser_time", "server"]


def call(day, q_start, q_time, outcome, ser_time, call_type="PS"):
    return ["AA0101", "1", "0", "1", call_type, day, q_start, q_start, "0",
            q_start, q_start, str(q_time), outcome, q_start, q_start, str(ser_time),
            "DORIT" if outcome == "AGENT" else "NO_SERVER"]


def write_month(folder: Path, name: str, rows):
    lines = ["\t".join(HEADER)] + ["\t".join(r) for r in rows]
    (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def sample(tmp_path):
    # 990905 = Sunday, 990901 = Wednesday (working days); 990903 = Friday, 990904 = Saturday
    rows = []
    for day in ("990905", "990901"):
        rows += [call(day, "10:05:00", 30, "AGENT", 180) for _ in range(4)]
        rows += [call(day, "15:10:00", 20, "AGENT", 200) for _ in range(2)]
        rows += [call(day, "10:20:00", 60, "AGENT", 400, "IN") for _ in range(2)]
        rows += [call(day, "10:30:00", 120, "HANG", 0)]
    rows += [call("990903", "10:05:00", 30, "AGENT", 180) for _ in range(50)]   # Friday: must not count
    rows += [call("990904", "21:05:00", 30, "AGENT", 180) for _ in range(50)]   # Saturday: must not count
    rows += [call("990905", "0:00:00", 0, "HANG", 0)]                         # hung up before the queue
    write_month(tmp_path, "sep1999.txt", rows)
    return tmp_path


def test_dates_are_read_in_1999():
    assert cal._parse_date("990901") == date(1999, 9, 1)
    assert cal._parse_date("990101").strftime("%A") == "Friday"   # 1 Jan 1999 was a Friday


def test_only_sunday_to_thursday_are_working_days():
    week = {d: cal._working_day(cal._parse_date(d)) for d in
            ("990905", "990906", "990907", "990908", "990909", "990903", "990904")}
    assert week == {"990905": True, "990906": True, "990907": True, "990908": True,
                    "990909": True, "990903": False, "990904": False}


def test_arrival_curve_ignores_fridays_and_saturdays(sample):
    result = cal.calibrate(sample, expected_files=None)
    arrivals = result["arrivals"]
    assert arrivals["working_day_count"] == 2
    raw = dict(zip(range(8, 20), arrivals["raw_calls_per_hour"]))
    assert raw[10] == pytest.approx(7.0)     # (4 PS + 2 IN + 1 hang-up) per working day; Friday's 50 left out
    assert raw[15] == pytest.approx(2.0)
    assert sum(arrivals["calls_per_hour"]) == pytest.approx(1050.0)


def test_service_and_patience(sample):
    result = cal.calibrate(sample, expected_files=None)
    assert result["service"]["technical"]["mean_minutes"] == pytest.approx(400 / 60)
    # patience = all queue waiting time / hang-ups (2 hang-ups, each after 120 s)
    waiting = 2 * (4 * 30 + 2 * 20 + 2 * 60 + 120) + 50 * 30 + 50 * 30
    assert result["patience"]["minutes"]["general"] == pytest.approx(waiting / 2 / 60)


def test_roster_leaves_room_above_the_erlang_peak(sample):
    roster = cal.calibrate(sample, expected_files=None)["roster"]
    total = roster["general_only"] + roster["general_technical"] + roster["general_complaint"]
    peak = roster["erlang_peak"]
    assert peak["total"] <= 0.8 * total
    assert peak["technical"] <= 0.8 * roster["general_technical"]
    assert peak["complaint"] <= 0.8 * roster["general_complaint"]


def test_roster_only_mode_keeps_everything_else(tmp_path):
    original = json.loads((ROOT / "workforce" / "calibration.json").read_text(encoding="utf-8"))
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    import subprocess
    import sys
    subprocess.run([sys.executable, str(ROOT / "scripts" / "calibrate_workforce.py"), "--roster-only",
                    "--output", str(path)], check=True, capture_output=True)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert {k: v for k, v in updated.items() if k != "roster"} == {k: v for k, v in original.items() if k != "roster"}
    assert updated["roster"]["erlang_peak"]["total"] <= 0.8 * sum(
        updated["roster"][k] for k in ("general_only", "general_technical", "general_complaint"))
