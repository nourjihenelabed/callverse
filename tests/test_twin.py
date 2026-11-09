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
    report = twin.compare(twin.measure(HALF), days=range(20_000, 20_002), with_rl=False)
    names = list(report["results"])
    assert len(names) == 4
    base, ignored, planned, better = (report["results"][n] for n in names)
    assert ignored["calls"] > base["calls"]                          # the callbacks arrive
    assert planned["sla"] > ignored["sla"]                           # planning for them restores the service
    assert better["total_cost_eur"] < planned["total_cost_eur"]      # a better advisor costs less


class ZeroCorrection:
    """Stands in for a trained PPO model: always 'keep the Erlang C plan' (correction 0, priority off)."""

    def predict(self, observation, deterministic=True):
        return [3, 3, 3, 0], None


def test_callback_share_is_an_assumption_you_can_change():
    measured = twin.measure(HALF)
    none = twin.twin_load(measured, callback_share=0.0)
    all_ = twin.twin_load(measured, callback_share=1.0)
    assert none["volume_factor"] < twin.twin_load(measured)["volume_factor"] < all_["volume_factor"]


def test_old_forecast_shows_the_policy_the_bank_forecast():
    from workforce.simulation import CallCenter

    measured = twin.measure(HALF)
    base, floor = CenterConfig(), twin.twin_config(twin.measure(HALF))

    class Spy:
        def decide(self, snapshot):
            self.snapshot = snapshot
            return {"general": 6, "technical": 4, "complaint": 2}

    spy = Spy()
    CallCenter(twin.OldForecast(spy, base, floor), seed=1, config=floor).run_until(130)
    hour = int(spy.snapshot.minute // 60)
    assert spy.snapshot.calls_next_hour == pytest.approx(base.calls_per_hour[hour])
    shares = {q: v / sum(spy.snapshot.expected_calls.values()) for q, v in spy.snapshot.expected_calls.items()}
    assert shares == pytest.approx(base.call_mix)                    # the bank mix, not the twin's
    assert measured["fcr"] < 1


def test_comparison_adds_the_rl_agents_when_they_exist(monkeypatch):
    pytest.importorskip("gymnasium")
    from workforce.env import RLPolicy

    monkeypatch.setattr(twin, "load_agent", lambda path, config: RLPolicy(ZeroCorrection(), config))
    report = twin.compare(twin.measure(HALF), days=range(20_000, 20_002))
    names = list(report["results"])
    assert names[-2:] == ["RL agent trained on bank data (does not know)", "RL agent retrained on the twin"]
    # an agent that never corrects anything does what its plan does
    assert report["results"][names[-2]]["calls"] == report["results"][names[1]]["calls"]


def test_the_twin_agent_is_not_mixed_with_the_bank_agents():
    assert not twin.TWIN_MODEL_PATH.name.startswith("ppo_staffing")   # compare.py globs ppo_staffing*.zip
