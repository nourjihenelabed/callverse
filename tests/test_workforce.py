"""Checks the pretend call center (workforce/simulation.py) and the baseline (workforce/manager.py).

No AI model and no database needed. Runs in a few seconds.
"""

import pytest

from workforce.manager import FixedRulePolicy, FixedStaffPolicy, advisors_needed, erlang_c
from workforce.simulation import QUEUES, CallCenter, CenterConfig, Policy, simulate_days


def test_erlang_c_matches_the_textbook_value():
    assert erlang_c(12, 10.0) == pytest.approx(0.449, abs=0.001)
    assert erlang_c(5, 6.0) == 1.0                    # not enough advisors: everyone waits


def test_more_calls_need_more_advisors():
    assert advisors_needed(120, 5, 20, 0.8) > advisors_needed(60, 5, 20, 0.8)
    assert advisors_needed(0, 5, 20, 0.8) == 0


def test_same_seed_gives_the_same_day():
    a = CallCenter(FixedRulePolicy(), seed=7).run().kpis()
    b = CallCenter(FixedRulePolicy(), seed=7).run().kpis()
    assert a == b


def test_every_call_is_counted_once():
    result = CallCenter(FixedRulePolicy(), seed=3).run()
    for q in QUEUES:
        assert result.answered[q] + result.abandoned[q] <= result.offered[q]
    total = sum(result.answered.values()) + sum(result.abandoned.values()) + result.unanswered_at_close
    assert total == sum(result.offered.values())


def test_kpis_stay_in_range():
    kpis = CallCenter(FixedRulePolicy(), seed=5).run().kpis()
    assert 0 <= kpis["sla"] <= 1
    assert 0 <= kpis["abandonment_rate"] <= 1
    assert 0 < kpis["occupancy"] <= 1
    assert 1 <= kpis["satisfaction"] <= 5
    assert kpis["calls"] > 800                         # about 1,050 calls expected per day


def test_nobody_on_duty_means_everyone_hangs_up():
    kpis = CallCenter(FixedStaffPolicy({}), seed=2).run().kpis()
    assert kpis["answered"] == 0
    assert kpis["staff_cost_eur"] == 0
    assert kpis["abandonment_rate"] > 0.95


def test_more_staff_means_shorter_waits_but_higher_cost():
    small = simulate_days(FixedStaffPolicy({"general": 5, "technical": 4, "complaint": 3}), range(3))
    large = simulate_days(FixedStaffPolicy({"general": 9, "technical": 9, "complaint": 6}), range(3))
    assert large["avg_wait_s"] < small["avg_wait_s"]
    assert large["abandonment_rate"] < small["abandonment_rate"]
    assert large["staff_cost_eur"] > small["staff_cost_eur"]


def test_baseline_meets_its_service_target():
    kpis = simulate_days(FixedRulePolicy(), range(5))
    assert kpis["sla"] >= CenterConfig().sla_target


def test_skills_limit_who_can_work_where():
    class AskTooMany(Policy):
        def decide(self, snapshot):
            assert snapshot.staffed["complaint"] <= 6   # only 6 advisors have the complaint skill
            return {"complaint": 50}

    CallCenter(AskTooMany(), seed=1).run()


def test_policy_sees_the_center():
    seen = []

    class Watcher(Policy):
        def decide(self, snapshot):
            seen.append(snapshot)
            return {"general": 9, "technical": 9, "complaint": 6}

    CallCenter(Watcher(), seed=1).run()
    assert len(seen) == (12 * 60 + 60) // 15          # every 15 minutes, opening hours + overtime
    assert seen[0].hour == 8 and seen[8].hour == 10
    assert seen[8].calls_next_hour == 120
