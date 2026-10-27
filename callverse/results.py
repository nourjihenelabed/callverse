"""Saves every finished call and its review, and computes the KPIs the dashboard shows.

The tables live in their own MetaData, not in db.py: scripts/seed_data.py drops and rebuilds
the db.py tables, and that must never erase the call results.

- calls: one row per call (who, why, mood, outcome, satisfaction, complaint decision, review)
- call_scores: one row per call and criterion (score and explanation)

Usage:
    from callverse.results import save_call, load_calls, summary
    save_call(done, run_id="night-1", seconds=95.0)
    print(summary(load_calls(run_id="night-1")))

Plain Python only (no pandas), so it works on every computer of the team.
"""

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text,
                        delete, insert, select)

from callverse.db import get_engine
from callverse.state import QUALITY_CRITERIA, CallState

results_metadata = MetaData()

calls = Table(
    "calls",
    results_metadata,
    Column("call_id", String(60), primary_key=True),
    Column("run_id", String(40), index=True),                   # a batch of calls, e.g. "night-1"
    Column("finished_at", DateTime, index=True),
    Column("seconds", Float),                                   # how long the AI workers took
    Column("customer_id", String(20), index=True),
    Column("persona", String(30)),
    Column("call_reason", String(40)),
    Column("queue", String(20)),                                # general, technical, complaint
    Column("intent", String(60)),
    Column("is_complaint", Boolean),
    Column("outcome", String(20)),
    Column("satisfaction", Integer),
    Column("turns", Integer),
    Column("complaint_category", String(30)),
    Column("complaint_severity", Integer),
    Column("compensation_cents", Integer),
    Column("escalated", Boolean),
    Column("ticket_id", String(20)),
    Column("wants_to_leave", Boolean),                          # added with the retention route
    Column("retention_offer", Text),                            # the supervisor's offer, if any
    Column("retention_supervisor", Boolean),
    Column("overall", Float),
    Column("escalation_correct", Boolean),
    Column("recommendations", Text),                            # JSON list
    Column("messages", Text),                                   # JSON list of {"role", "content"}
    Column("tools_used", Text),                                 # JSON list of {"name", "arguments"}
)

call_scores = Table(
    "call_scores",
    results_metadata,
    Column("call_id", String(60), ForeignKey("calls.call_id", ondelete="CASCADE"), primary_key=True),
    Column("criterion", String(20), primary_key=True),          # one of QUALITY_CRITERIA
    Column("score", Integer),
    Column("explanation", Text),
)


def create_tables(engine=None) -> None:
    """Creates the two tables if they are missing, and adds columns added later (the saved calls stay).
    Never deletes anything."""
    from sqlalchemy import inspect, text

    engine = engine or get_engine()
    results_metadata.create_all(engine)
    existing = {c["name"] for c in inspect(engine).get_columns("calls")}
    missing = [c for c in calls.columns if c.name not in existing]
    if missing:
        with engine.begin() as conn:
            for column in missing:
                kind = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f"ALTER TABLE calls ADD COLUMN {column.name} {kind}"))


def queue_of(call_reason: str) -> str:
    from callverse.agents.client.scenarios import REASONS

    reason = REASONS.get(call_reason)
    return reason.queue if reason else ("complaint" if call_reason.startswith("complaint") else "general")


# ---------------------------------------------------------------- saving

def save_call(done: CallState, run_id: str, seconds: float, engine=None) -> None:
    """Saves one finished call. Saving the same call_id again replaces it."""
    engine = engine or get_engine()
    create_tables(engine)
    r, q = done.complaint_result, done.quality_report
    row = {
        "call_id": done.call_id, "run_id": run_id, "finished_at": datetime.now(), "seconds": round(seconds, 1),
        "customer_id": done.customer_id, "persona": done.persona, "call_reason": done.call_reason,
        "queue": queue_of(done.call_reason), "intent": done.intent, "is_complaint": done.is_complaint,
        "outcome": done.outcome, "satisfaction": done.satisfaction, "turns": done.turn,
        "complaint_category": r.category if r else None, "complaint_severity": r.severity if r else None,
        "compensation_cents": r.compensation_cents if r else None, "escalated": r.escalate if r else None,
        "ticket_id": r.ticket_id if r else None,
        "wants_to_leave": done.wants_to_leave,
        "retention_offer": done.retention.offer if done.retention else None,
        "retention_supervisor": done.retention.supervisor if done.retention else None,
        "overall": q.overall if q else None, "escalation_correct": q.escalation_correct if q else None,
        "recommendations": json.dumps(q.recommendations if q else [], ensure_ascii=False),
        "messages": json.dumps([m.model_dump() for m in done.messages], ensure_ascii=False),
        "tools_used": json.dumps([t.model_dump() for t in done.tools_used], ensure_ascii=False),
    }
    scores = [{"call_id": done.call_id, "criterion": name, "score": s.score, "explanation": s.explanation}
              for name, s in (q.scores.items() if q else [])]
    with engine.begin() as conn:
        conn.execute(delete(call_scores).where(call_scores.c.call_id == done.call_id))
        conn.execute(delete(calls).where(calls.c.call_id == done.call_id))
        conn.execute(insert(calls), [row])
        if scores:
            conn.execute(insert(call_scores), scores)


def saved_call_ids(run_id: str, engine=None) -> set[str]:
    engine = engine or get_engine()
    create_tables(engine)
    with engine.connect() as conn:
        return set(conn.execute(select(calls.c.call_id).where(calls.c.run_id == run_id)).scalars())


# ---------------------------------------------------------------- loading

def load_calls(run_id: str | None = None, engine=None) -> list[dict]:
    """Every saved call (or one run's) as a dict, with "scores" = {criterion: {"score", "explanation"}}."""
    engine = engine or get_engine()
    create_tables(engine)
    query = select(calls).order_by(calls.c.finished_at)
    if run_id:
        query = query.where(calls.c.run_id == run_id)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query)]
        grades = defaultdict(dict)
        for g in conn.execute(select(call_scores)):
            grades[g.call_id][g.criterion] = {"score": g.score, "explanation": g.explanation}
    for row in rows:
        for key in ("recommendations", "messages", "tools_used"):
            row[key] = json.loads(row[key] or "[]")
        row["scores"] = grades.get(row["call_id"], {})
    return rows


def run_ids(engine=None) -> list[str]:
    engine = engine or get_engine()
    create_tables(engine)
    with engine.connect() as conn:
        return list(conn.execute(select(calls.c.run_id).distinct().order_by(calls.c.run_id)).scalars())


# ---------------------------------------------------------------- KPIs

def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def _share(rows: list[dict], test) -> float | None:
    return round(sum(1 for r in rows if test(r)) / len(rows), 3) if rows else None


def summary(rows: list[dict]) -> dict:
    """The headline numbers of a set of calls."""
    complaints = [r for r in rows if r["is_complaint"]]
    checked = [r for r in complaints if r["escalation_correct"] is not None]
    leaving = [r for r in rows if r.get("wants_to_leave")]
    return {
        "calls": len(rows),
        "resolved_rate": _share(rows, lambda r: r["outcome"] == "resolved"),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "avg_satisfaction": _mean(r["satisfaction"] for r in rows),
        "avg_quality": _mean(r["overall"] for r in rows),
        "criteria": {c: _mean(r["scores"].get(c, {}).get("score") for r in rows) for c in QUALITY_CRITERIA},
        "complaint_rate": _share(rows, lambda r: r["is_complaint"]),
        "escalation_rate": _share(complaints, lambda r: r["escalated"]),
        "escalation_accuracy": _share(checked, lambda r: r["escalation_correct"]),
        "avg_compensation_eur": _mean((r["compensation_cents"] or 0) / 100 for r in complaints),
        "wanted_to_leave": len(leaving),
        "kept_rate": _share(leaving, lambda r: r["outcome"] in ("resolved", "escalated")),
        "offer_rate": _share(leaving, lambda r: bool(r.get("retention_offer"))),
        "avg_turns": _mean(r["turns"] for r in rows),
        "avg_seconds": _mean(r["seconds"] for r in rows),
    }


def by(rows: list[dict], key: str) -> dict[str, dict]:
    """The same numbers for each persona, queue, reason... e.g. by(rows, "persona")."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {name: summary(group) for name, group in sorted(groups.items(), key=lambda g: str(g[0]))}


def daily(rows: list[dict]) -> dict[str, dict]:
    """The KPIs of each day calls were run."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["finished_at"].date().isoformat()].append(r)
    return {day: summary(group) for day, group in sorted(groups.items())}


def common_tips(rows: list[dict], top: int = 5) -> list[tuple[str, int]]:
    """The reviewer's most frequent recommendations: what the advisor should improve first."""
    return Counter(t for r in rows for t in r["recommendations"]).most_common(top)
