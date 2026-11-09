"""The studio streams a real call through the graph (with the offline fakes of conftest.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio"))

import server  # noqa: E402

from callverse.state import CallState, Message  # noqa: E402


def test_a_call_streams_agent_starts_updates_and_the_finished_case_file():
    state = CallState(call_id="s1", customer_id="0000-TEST", persona="calm", call_reason="check_invoice")
    events = list(server.call_events(state, save=False))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and "error" not in kinds
    started = [e["node"] for e in events if e["type"] == "start"]
    assert started[0] == "client" and "advisor" in started and started[-1] == "quality"
    updated = [e["node"] for e in events if e["type"] == "update"]
    assert updated == started                       # every agent that starts reports what it changed
    final = events[-1]["state"]
    assert final["outcome"] and final["messages"] and final["quality_report"]["overall"]
    assert events[-1]["saved"] is False


def test_errors_are_reported_instead_of_crashing(monkeypatch):
    import callverse.graph as graph

    def broken():
        raise RuntimeError("model 'llama3.1:8b' not found")
    monkeypatch.setattr(graph, "build_graph", broken)
    state = CallState(call_id="s2", customer_id="0000-TEST", persona="calm", call_reason="check_invoice")
    events = list(server.call_events(state, save=False))
    assert events[-1]["type"] == "error" and "not found" in events[-1]["message"]


def test_jsonable_handles_models_and_dates():
    from datetime import datetime
    out = server.jsonable({"m": Message(role="client", content="Bonjour"), "at": datetime(2026, 1, 2)})
    assert out == {"m": {"role": "client", "content": "Bonjour"}, "at": "2026-01-02T00:00:00"}


def test_a_workforce_day_compares_the_policies_on_the_same_calls():
    day = server.workforce_day("network_outage", 10_000)
    names = list(day["policies"])
    assert names[:3] == ["Erlang C plan", "Reactive rule", "Margin rule"]
    calls = {day["policies"][n]["kpis"]["calls"] for n in names}
    assert max(calls) - min(calls) < 30                     # same arrivals (hang-ups differ slightly)
    timeline = day["policies"]["Erlang C plan"]["timeline"]
    assert len(timeline) == 52 and set(timeline[0]["staffed"]) == {"general", "technical", "complaint"}
