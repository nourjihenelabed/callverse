"""Checks the scoreboard: the charts (dashboard/charts.py), the HTML report (scripts/build_report.py)
and the three pages of the Streamlit app (dashboard/app.py, skipped if Streamlit is not installed).

No Ollama and no PostgreSQL: calls run with the offline fakes and are saved in a temporary SQLite file.
"""

import importlib.util
from pathlib import Path

import pytest

from callverse import results
from callverse.graph import run_call
from callverse.state import CallState
from test_results import engine  # noqa: F401  (the SQLite fixture)

pytest.importorskip("plotly")
from dashboard import charts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts" / "build_report.py")
build_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_report)

POLICIES = charts.WFM_POLICIES
KPIS = {"sla": 0.82, "abandonment_rate": 0.04, "avg_wait_s": 9.0, "occupancy": 0.8, "staff_cost_eur": 3000,
        "satisfaction": 4.1, "total_cost_eur": 3500}
COMPARISON = {s: {**{p: {**KPIS, "total_cost_eur": 3500 + 100 * i} for i, p in enumerate(POLICIES)},
                  "rl_spread_between_agents": {k: 0.0 for k in KPIS}, "agents": 3}
              for s in ("normal", "network_outage")}


TWIN = {"run": "demo", "target_fcr": 0.8, "callback_share": 0.5,
        "measured": {"calls": 30, "fcr": 0.5, "queues": {q: {"calls": 10, "fcr": 0.5, "to_complaint": 0.1,
                                                              "avg_turns": 3} for q in ("general", "technical", "complaint")}},
        "load": {"volume_factor": 1.25, "call_mix": {"general": 0.5, "technical": 0.3, "complaint": 0.2},
                 "repeat_factor": {"general": 1.25, "technical": 1.25, "complaint": 1.25}},
        "results": {name: {"calls": 1000, "sla": 0.85, "abandonment_rate": 0.03, "staff_cost_eur": 4000,
                           "total_cost_eur": cost} for name, cost in
                    [("As calibrated", 6000), ("Ignored", 8500), ("Planned", 7300), ("Better advisor", 6800)]}}


def test_twin_tables_and_chart():
    assert len(charts.twin_rows(TWIN)) == 4 and charts.twin_costs(TWIN).data
    assert charts.twin_measured_rows(TWIN)[0]["Resolved first time"] == "50%"


@pytest.fixture
def saved(engine):  # noqa: F811
    for n, (reason, persona) in enumerate([("check_invoice", "calm"), ("complaint_billing", "dissatisfied")]):
        done = run_call(CallState(call_id=f"d-{n}", customer_id="0001-AAAAA", persona=persona, call_reason=reason))
        results.save_call(done, run_id="demo", seconds=30)
    return results.load_calls()


def test_every_chart_is_drawn_from_saved_calls(saved):
    for fig in (charts.criteria(saved), charts.outcomes(saved), charts.grouped(saved, "persona", "x"),
                charts.over_time(saved), charts.call_grades(saved[0])):
        assert fig.data, fig.layout.title.text
    assert list(charts.criteria(saved).data[0].x) == list(charts.CRITERIA_LABELS.values())


def test_workforce_summary_ranks_the_policies():
    lines = charts.wfm_summary(COMPARISON)
    assert [line["Policy"] for line in lines] == list(POLICIES)
    assert lines[0]["Worst gap to the best (€/day)"] == 0 and lines[3]["Worst gap to the best (€/day)"] == 300
    assert lines[0]["Scenarios meeting the SLA"] == "2/2"
    assert charts.wfm_metric(COMPARISON, "sla", "SLA", percent=True).data


def test_missing_workforce_results_are_not_an_error(tmp_path):
    assert charts.load_comparison(tmp_path / "comparison.json") is None


def test_report_has_the_calls_their_reasons_and_works_offline(saved, monkeypatch):
    monkeypatch.setattr(charts, "load_twin", lambda: TWIN)
    page = build_report.build(saved, COMPARISON, "Test")
    assert "what the AI calls change for staffing" in page
    assert page.startswith("<!doctype html>") and page.count("plotly.js") >= 1
    assert "Why these grades" in page and "Vérification automatique" in page
    assert saved[0]["messages"][0]["content"][:30] in page
    assert "Workforce manager" in page and "RL agent (PPO)" in page
    assert "src=\"https://cdn" not in page                       # the chart library is inside the page


def test_report_without_calls_says_what_to_run():
    assert "run_calls.py" in build_report.build([], None, "Empty", include_twin=False)


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="Streamlit not installed")
@pytest.mark.parametrize("page", ["Calls", "One call", "Workforce manager"])
def test_every_streamlit_page_runs(saved, monkeypatch, page):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(charts, "load_comparison", lambda: COMPARISON)
    monkeypatch.setattr(charts, "load_twin", lambda: TWIN)
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.title[0].value == page
