"""Checks the saved call results (callverse/results.py) and the batch runner (scripts/run_calls.py).

No Ollama and no PostgreSQL: whole calls run with the offline fakes of conftest.py,
and a temporary SQLite file replaces the database.
"""

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from callverse import results
from callverse.graph import run_call
from callverse.state import QUALITY_CRITERIA, CallState

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_calls", ROOT / "scripts" / "run_calls.py")
run_calls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_calls)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'results.db'}")
    monkeypatch.setattr(results, "get_engine", lambda: engine)
    return engine


def finished(call_id="r-1", reason="check_invoice", persona="calm") -> CallState:
    return run_call(CallState(call_id=call_id, customer_id="0001-AAAAA", persona=persona, call_reason=reason))


def test_a_saved_call_comes_back_whole(engine):
    done = finished()
    results.save_call(done, run_id="test", seconds=12.34)
    [row] = results.load_calls(run_id="test")

    assert row["call_id"] == "r-1" and row["queue"] == "general" and row["seconds"] == 12.3
    assert row["outcome"] == done.outcome and row["satisfaction"] == done.satisfaction
    assert [m["role"] for m in row["messages"]] == [m.role for m in done.messages]
    assert set(row["scores"]) == set(QUALITY_CRITERIA)
    assert row["scores"]["satisfaction"]["explanation"]
    assert row["overall"] == done.quality_report.overall


def test_a_complaint_keeps_the_specialist_decision(engine):
    done = finished("r-2", "complaint_billing")
    results.save_call(done, run_id="test", seconds=1)
    [row] = results.load_calls()
    assert row["is_complaint"] and row["queue"] == "complaint"
    assert row["complaint_category"] == done.complaint_result.category
    assert row["ticket_id"] == done.complaint_result.ticket_id
    assert row["escalation_correct"] is not None


def test_saving_again_replaces_the_call(engine):
    done = finished()
    results.save_call(done, run_id="test", seconds=1)
    results.save_call(done, run_id="test", seconds=2)
    assert [r["seconds"] for r in results.load_calls()] == [2]


def test_reseeding_the_customers_never_erases_the_results(engine):
    results.save_call(finished(), run_id="test", seconds=1)
    from callverse.db import metadata
    metadata.drop_all(engine)                 # what scripts/seed_data.py does
    metadata.create_all(engine)
    assert len(results.load_calls()) == 1


def test_summary_and_groups():
    rows = [
        {"outcome": "resolved", "satisfaction": 5, "overall": 4.5, "is_complaint": False, "escalated": None,
         "escalation_correct": None, "compensation_cents": None, "turns": 2, "seconds": 60.0, "persona": "calm",
         "scores": {"accuracy": {"score": 5}}, "recommendations": ["A"]},
        {"outcome": "abandoned", "satisfaction": 1, "overall": 2.5, "is_complaint": True, "escalated": True,
         "escalation_correct": True, "compensation_cents": 500, "turns": 6, "seconds": 120.0,
         "persona": "impatient", "scores": {"accuracy": {"score": 3}}, "recommendations": ["A", "B"]},
    ]
    s = results.summary(rows)
    assert s["calls"] == 2 and s["resolved_rate"] == 0.5 and s["avg_satisfaction"] == 3
    assert s["avg_quality"] == 3.5 and s["criteria"]["accuracy"] == 4 and s["criteria"]["empathy"] is None
    assert s["complaint_rate"] == 0.5 and s["escalation_accuracy"] == 1.0 and s["avg_compensation_eur"] == 5
    assert set(results.by(rows, "persona")) == {"calm", "impatient"}
    assert results.common_tips(rows)[0] == ("A", 2)
    assert results.summary([])["calls"] == 0


def test_batch_saves_every_call_and_continues_after_a_stop(engine):
    from callverse.agents.client import scenarios

    def draw(rng, call_id):        # the real drawing, on a made-up account (no database)
        reason = scenarios.pick_reason(rng, {"codes": ["box_fibre", "forfait_mobile"]})
        return CallState(call_id=call_id, customer_id="0001-AAAAA", persona="calm", call_reason=reason)

    first = run_calls.run(3, "night", seed=5, draw=draw)
    assert first == {"done": 3, "skipped": 0, "failed": 0}
    reasons = [r["call_reason"] for r in results.load_calls("night")]

    again = run_calls.run(5, "night", seed=5, draw=draw)        # same run: the 3 saved calls are skipped
    assert again == {"done": 2, "skipped": 3, "failed": 0}
    rows = results.load_calls("night")
    assert [r["call_reason"] for r in rows][:3] == reasons and len(rows) == 5


def test_a_crashing_call_does_not_stop_the_batch(engine):
    def draw(rng, call_id):
        return CallState(call_id=call_id, customer_id="0001-AAAAA", persona="calm", call_reason="check_invoice")

    def sometimes_crash(state):
        if state.call_id.endswith("2"):
            raise RuntimeError("Ollama stopped")
        return run_call(state)

    counts = run_calls.run(3, "crash", seed=1, call=sometimes_crash, draw=draw)
    assert counts == {"done": 2, "skipped": 0, "failed": 1}


def test_old_tables_get_the_new_columns_and_keep_their_calls(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:            # a calls table made before the retention route
        conn.execute(text("CREATE TABLE calls (call_id VARCHAR(60) PRIMARY KEY, run_id VARCHAR(40), "
                          "finished_at DATETIME, recommendations TEXT, messages TEXT, tools_used TEXT)"))
        conn.execute(text("INSERT INTO calls (call_id, run_id, finished_at) VALUES ('old-1', 'night-1', '2026-10-26')"))
    results.create_tables(engine)
    rows = results.load_calls(engine=engine)
    assert [r["call_id"] for r in rows] == ["old-1"] and rows[0]["wants_to_leave"] is None


def test_retention_numbers_in_the_summary():
    base = {"satisfaction": 3, "overall": 4.0, "is_complaint": False, "escalated": None, "escalation_correct": None,
            "compensation_cents": None, "turns": 2, "seconds": 60.0, "scores": {}, "recommendations": []}
    rows = [{**base, "outcome": "resolved", "wants_to_leave": True, "retention_offer": "3 mois offerts"},
            {**base, "outcome": "refused", "wants_to_leave": True, "retention_offer": None},
            {**base, "outcome": "resolved", "wants_to_leave": False, "retention_offer": None}]
    s = results.summary(rows)
    assert s["wanted_to_leave"] == 2 and s["kept_rate"] == 0.5 and s["offer_rate"] == 0.5
