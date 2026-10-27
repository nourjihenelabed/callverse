"""Checks scripts/compare_runs.py: the statistics and the tables (no database needed)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "scripts" / "compare_runs.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def row(outcome, satisfaction, persona="calm", grade=4, leave=False, offer=None):
    scores = {c: {"score": grade} for c in ("relevance", "accuracy", "compliance", "empathy")}
    scores |= {"resolution": {"score": 5 if outcome == "resolved" else 1}, "satisfaction": {"score": satisfaction}}
    return {"outcome": outcome, "satisfaction": satisfaction, "overall": 3.5, "persona": persona, "scores": scores,
            "is_complaint": False, "escalated": None, "escalation_correct": None, "compensation_cents": None,
            "turns": 3, "seconds": 60.0, "recommendations": [], "wants_to_leave": leave, "retention_offer": offer}


def test_two_proportions_matches_the_textbook():
    # 52% vs 62% on 50 calls each: z = 1.01, p = 0.31
    assert cr.two_proportions(26, 50, 31, 50) == pytest.approx(0.31, abs=0.01)
    assert cr.two_proportions(10, 100, 40, 100) < 0.001
    assert cr.two_proportions(5, 10, 5, 10) == 1.0


def test_welch_sees_a_real_difference_and_not_a_tiny_one():
    assert cr.welch([1, 2, 1, 2] * 10, [4, 5, 4, 5] * 10) < 0.001
    assert cr.welch([3, 4] * 10, [3, 4, 4, 3] * 5) > 0.5
    assert cr.welch([3], [4]) is None


def test_quality_without_the_outcome_ignores_resolution_and_satisfaction():
    assert cr.quality_without_outcome(row("abandoned", 1, grade=4)) == 4


def test_tables():
    a = [row("resolved", 4)] * 20 + [row("abandoned", 1, "likely_to_churn")] * 5
    b = [row("resolved", 4)] * 20 + [row("resolved", 3, "likely_to_churn", leave=True, offer="3 mois offerts")] * 5
    lines = dict((m, (x, y, t)) for m, x, y, t in cr.compare(a, b))
    assert lines["Resolved"][0] == "20/25 (80%)" and lines["Resolved"][1] == "25/25 (100%)"
    moods = dict((m, (x, y)) for m, x, y in cr.by_mood(a, b))
    assert moods["likely_to_churn"] == ("0/5 resolved, satisfaction 1", "5/5 resolved, satisfaction 3")
    assert "5 wanted to leave, 100% got a supervisor's offer" in cr.retention_lines(b)
    assert "no customer flagged" in cr.retention_lines(a)
    text = cr.markdown("night-1", "night-2", a, b)
    assert "| Resolved | 20/25 (80%) | 25/25 (100%) |" in text and "### By customer mood" in text
