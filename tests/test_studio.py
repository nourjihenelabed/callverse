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


def test_a_traced_call_shows_the_functions_and_the_routing_then_puts_everything_back():
    import walkthrough

    import callverse.graph as graph
    from callverse.agents.complaint import steps
    before = (graph.after_advisor, list(steps.STEPS))
    state = CallState(call_id="s3", customer_id="0000-TEST", persona="dissatisfied", call_reason="complaint_billing")
    events = list(walkthrough.traced_events(server.call_events, state, save=False))
    assert events[-1]["type"] == "done"
    traces = [e for e in events if e["type"] == "trace"]
    funcs = [e["func"] for e in traces]
    # the complaint specialist's steps, in order, then its ticket
    assert [f for f in funcs if f in walkthrough.STEP_PROBES] == list(walkthrough.STEP_PROBES)
    assert "open_ticket" in funcs and "understand" in funcs and "judge" in funcs
    routes = [e["text"] for e in traces if e["kind"] == "route"]
    assert routes[0].startswith("→ advisor") and any(r.startswith("→ complaint:") for r in routes)
    # a trace comes after its agent starts: the first one belongs to the customer
    assert events.index(traces[0]) > next(i for i, e in enumerate(events) if e["type"] == "start")
    assert (graph.after_advisor, list(steps.STEPS)) == before          # nothing left wrapped


def test_only_one_call_is_traced_at_a_time():
    import walkthrough
    with walkthrough.tracing([]) as first:
        with walkthrough.tracing([]) as second:
            assert first is True and second is False


def test_every_file_and_function_named_in_the_walkthrough_exists():
    import ast
    import importlib

    import walkthrough
    root = Path(__file__).resolve().parent.parent

    def defined(file: str, name: str) -> bool:
        tree = ast.parse((root / file).read_text(encoding="utf-8"))
        return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree))

    for node in walkthrough.NODES.values():
        assert defined(node["file"], node["func"]), node
    for module, attr, _, file, _ in walkthrough.PROBES:
        assert callable(getattr(importlib.import_module(module), attr)), (module, attr)
        assert (root / file).exists(), file
    for name in walkthrough.STEP_PROBES:
        assert defined(walkthrough.STEPS_FILE, name), name


def test_the_scenarios_list_works_without_a_database(monkeypatch):
    import walkthrough

    def no_database(kind, demo):
        raise ConnectionError("no PostgreSQL")
    monkeypatch.setattr(walkthrough, "pick_customer", no_database)
    data = server.scenarios()
    assert len(data["scenarios"]) == len(walkthrough.SCENARIOS)
    for s in data["scenarios"]:
        assert s["customer"] == server.DEMO_CUSTOMER
        assert s["route"][0]["node"] == "client" and s["route"][-1]["node"] == "quality"
        assert all(step["node"] in data["nodes"] for step in s["route"])


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


MEASURED = {"calls": 50, "fcr": 0.68, "queues": {
    "general": {"calls": 27, "fcr": 0.7, "to_complaint": 0.1, "avg_turns": 4},
    "technical": {"calls": 15, "fcr": 0.6, "to_complaint": 0.2, "avg_turns": 5},
    "complaint": {"calls": 8, "fcr": 0.75, "to_complaint": 0.0, "avg_turns": 5}}}


def test_the_twin_view_turns_quality_into_workload(monkeypatch):
    monkeypatch.setattr(server, "twin_measured", lambda run: (MEASURED, "saved"))
    view = server.twin_view({"run": "night-2"})
    assert view["base"]["calls_per_day"] == 1050
    assert view["twin"]["calls_per_day"] > 1050                       # the repeat calls
    assert view["twin"]["mix"]["complaint"] > view["base"]["mix"]["complaint"]
    better = server.twin_view({"run": "night-2", "fcr": "0.95"})
    assert better["twin"]["calls_per_day"] < view["twin"]["calls_per_day"]   # a better advisor, fewer calls
    nobody = server.twin_view({"run": "night-2", "callback": "0"})
    assert nobody["twin"]["calls_per_day"] < view["twin"]["calls_per_day"]


def test_a_workforce_day_with_the_twin_workload(monkeypatch):
    monkeypatch.setattr(server, "twin_measured", lambda run: (MEASURED, "saved"))
    bank = server.workforce_day("normal", 10_000)
    day = server.workforce_day("normal", 10_000, {"workload": "twin", "run": "night-2"})
    assert day["workload"] == "twin" and day["twin"]["calls_per_day"] > 1050
    names = list(day["policies"])
    assert names[:2] == ["Erlang C plan, ignores the repeat calls", "Erlang C plan, uses them"]
    assert day["policies"][names[0]]["kpis"]["calls"] > bank["policies"]["Erlang C plan"]["kpis"]["calls"]


def test_no_twin_data_is_a_clear_message(monkeypatch):
    monkeypatch.setattr(server, "twin_measured", lambda run: (None, "none"))
    import pytest
    with pytest.raises(server.NoTwinData):
        server.twin_view({})
