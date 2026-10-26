"""Checks phase 3, the loop from the AI calls to the staffing (workforce/twin.py).

No Ollama and no database: the calls are made-up rows like those of callverse.results.load_calls.
"""

import pytest

from workforce import twin
from workforce.simulation import QUEUES, CenterConfig


def rows(queue, outcomes, complaint=0):
    return [{"queue": queue, "outcome": o, "turns": 3, "is_complaint": i < complaint} for i, o in enumerate(outcomes)]


GOOD = rows("general", ["resolved"] * 10) + rows("technical", ["resolved"] * 10) + rows("complaint", ["resolved"] * 10)
HALF = (rows("general", ["resolved", "abandoned"] * 5, complaint=2) + rows("technical", ["resolved", "refused"] * 5)
        + rows("complaint", ["escalated", "abandoned"] * 5))


def test_measure_counts_resolution_and_calls_that_became_complaints():
    m = twin.measure(HALF)
    assert m["calls"] == 30 and m["fcr"] == 0.5
    assert m["queues"]["general"] == {"calls": 10, "fcr": 0.5, "to_complaint": 0.2, "avg_turns": 3}
    assert m["queues"]["complaint"]["fcr"] == 0.5          # escalated counts as handled


def test_perfect_calls_change_nothing():
    load = twin.twin_load(twin.measure(GOOD))
    assert load["volume_factor"] == 1
    assert load["call_mix"] == pytest.approx(CenterConfig().call_mix, abs=1e-4)


def test_unresolved_customers_call_back():
    load = twin.twin_load(twin.measure(HALF))
    assert load["repeat_factor"] == {q: 1.25 for q in QUEUES}        # 50% unresolved x 50% call back
    assert load["volume_factor"] == pytest.approx(1.25)
    assert load["call_mix"]["complaint"] > CenterConfig().call_mix["complaint"]   # 20% of general moved


def test_too_few_calls_in_a_queue_are_not_trusted():
    few = rows("general", ["abandoned"] * 3) + rows("technical", ["resolved"] * 10) + rows("complaint", ["resolved"] * 10)
    assert twin.twin_load(twin.measure(few))["repeat_factor"]["general"] == 1.0


def test_twin_config_keeps_the_real_durations_and_team():
    base = CenterConfig()
    config = twin.twin_config(twin.measure(HALF), base)
    assert config.mean_handle_minutes == base.mean_handle_minutes and config.roster == base.roster
    assert sum(config.calls_per_hour) == pytest.approx(1.25 * sum(base.calls_per_hour))
    assert sum(config.call_mix.values()) == pytest.approx(1, abs=1e-3)


def test_what_if_scenario_uses_the_target_resolution():
    measured = twin.measure(HALF)
    better = twin.twin_load(measured, fcr={q: 0.8 for q in QUEUES})
    assert better["volume_factor"] < twin.twin_load(measured)["volume_factor"]


def test_comparison_on_the_same_days():
    report = twin.compare(twin.measure(HALF), days=range(20_000, 20_002))
    names = list(report["results"])
    assert len(names) == 4
    base, ignored, planned, better = (report["results"][n] for n in names)
    assert ignored["calls"] > base["calls"]                          # the callbacks arrive
    assert planned["sla"] > ignored["sla"]                           # planning for them restores the service
    assert better["total_cost_eur"] < planned["total_cost_eur"]      # a better advisor costs less
