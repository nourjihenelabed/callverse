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


def test_phase_one_calibration_is_loaded():
    config = CenterConfig()
    assert set(config.mean_handle_minutes) == {"general", "technical", "complaint"}
    assert all(value > 0 for value in config.mean_handle_minutes.values())
    assert config.mean_patience_minutes["general"] > 0
    assert config.handle_spread > 0
    assert config.forecast_error == pytest.approx(0.2)
    assert len(config.calls_per_hour) == 12
    assert sum(count for count, _ in config.roster) == 21


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
    assert seen[8].calls_next_hour == CenterConfig().calls_per_hour[2]


def test_normal_day_has_no_surprise_calls_and_peaks_add_calls():
    from workforce.simulation import SCENARIOS
    assert SCENARIOS["normal"] == ()
    normal = CallCenter(FixedRulePolicy(), seed=4).run().kpis()
    for scenario in ("network_outage", "billing_day", "afternoon_promo"):
        peak = CallCenter(FixedRulePolicy(), seed=4, scenario=scenario).run().kpis()
        assert peak["calls"] > normal["calls"]
        assert peak["abandonment_rate"] > normal["abandonment_rate"]   # the fixed plan suffers


def test_reactive_rule_adds_advisors_only_when_customers_wait():
    from workforce.manager import ReactiveRulePolicy

    seen = []

    class Spy(ReactiveRulePolicy):
        def decide(self, snapshot):
            wanted = super().decide(snapshot)
            seen.append((snapshot, wanted, FixedRulePolicy.decide(self, snapshot)))
            return wanted

    CallCenter(Spy(), seed=1, scenario="billing_day").run()
    for snapshot, wanted, planned in seen:
        for q in QUEUES:
            extra = wanted[q] - planned[q]
            assert 0 <= extra <= 5
            assert extra == 0 or snapshot.waiting[q] > 0


def test_advisors_called_in_stay_at_least_one_hour():
    class OnOff(Policy):
        """Calls 5 general advisors in at 8:00, then asks to send everyone home."""
        def decide(self, snapshot):
            return {"general": 5} if snapshot.minute == 0 else {}

    center = CallCenter(OnOff(), seed=1)
    center.run_until(16)                     # after the 8:15 decision: still on duty
    assert sum(1 for a in center.assignment if a) == 5
    center.run_until(61)                     # after 9:00: the hour is over, they can leave
    assert sum(1 for a in center.assignment if a) == 0


def test_moving_to_another_queue_takes_set_up_time():
    class Move(Policy):
        def decide(self, snapshot):
            # technicians first work on technical calls, then everyone is needed on general calls
            return {"technical": 9} if snapshot.minute < 60 else {"general": 18}

    kpis = CallCenter(Move(), seed=1).run().kpis()
    assert kpis["queue_switches"] > 0


def test_priority_lets_specialists_on_general_take_complaints():
    class GeneralOnly(Policy):
        def __init__(self, priority):
            self.priority = priority

        def decide(self, snapshot):
            # all 30 advisors on general calls (complaint-skilled ones too); nobody on the complaint queue
            return {"general": 30, "priority": self.priority}

    without = CallCenter(GeneralOnly(False), seed=2).run().kpis()
    with_priority = CallCenter(GeneralOnly(True), seed=2).run().kpis()
    assert without["per_queue"]["complaint"]["abandonment_rate"] > 0.9   # nobody answers complaints
    assert with_priority["per_queue"]["complaint"]["abandonment_rate"] < 0.5


def test_snapshot_shows_calls_received_against_the_forecast():
    snaps = []

    class Spy(FixedRulePolicy):
        def decide(self, snapshot):
            snaps.append(snapshot)
            return super().decide(snapshot)

    CallCenter(Spy(), seed=1, scenario="network_outage").run()
    during = [s for s in snaps if 11 <= s.hour < 13]            # in the middle of the outage
    received = sum(s.last_calls["technical"] for s in during)
    expected = sum(s.expected_calls["technical"] for s in during)
    assert received > 1.5 * expected


def test_quiet_day_has_fewer_calls_and_normal_day_is_unchanged():
    normal = CallCenter(FixedRulePolicy(), seed=5).run().kpis()
    quiet = CallCenter(FixedRulePolicy(), seed=5, scenario="quiet_day").run().kpis()
    assert quiet["calls"] < 0.8 * normal["calls"]


def test_margin_rule_adds_its_margin_and_priority():
    from workforce.manager import MarginRulePolicy

    seen = []

    class Spy(MarginRulePolicy):
        def decide(self, snapshot):
            wanted = super().decide(snapshot)
            seen.append((wanted, FixedRulePolicy.decide(self, snapshot)))
            return wanted

    CallCenter(Spy(), seed=1).run()
    for wanted, planned in seen:
        assert wanted["general"] == planned["general"] + 4 and wanted["priority"] is True
