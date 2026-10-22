"""Floor manager: learns where to place workers by practicing many pretend days.

This file has the policies that decide how many advisors work on each queue:
- FixedRulePolicy: the classic baseline. A staffing plan computed in advance,
  hour by hour, with the Erlang C formula that real call centers use. It never
  reacts to what happens during the day.
- (next step) the RL agent, trained with Stable-Baselines3 on the same simulation.
  Both are compared on the same days with workforce.simulation.simulate_days().
"""

import math

from workforce.simulation import QUEUES, CenterConfig, Policy, Snapshot


# ---------------------------------------------------------------- Erlang C

def erlang_c(advisors: int, traffic: float) -> float:
    """Probability that a call has to wait, with `traffic` Erlangs and `advisors` agents."""
    if advisors <= traffic:
        return 1.0
    term, total = 1.0, 1.0
    for k in range(1, advisors):
        term *= traffic / k
        total += term
    last = term * traffic / advisors * advisors / (advisors - traffic)
    return last / (total + last)


def service_level(advisors: int, traffic: float, handle_minutes: float, target_seconds: float) -> float:
    """Share of calls answered within target_seconds."""
    if advisors <= traffic:
        return 0.0
    wait_chance = erlang_c(advisors, traffic)
    return 1 - wait_chance * math.exp(-(advisors - traffic) * (target_seconds / 60) / handle_minutes)


def advisors_needed(calls_per_hour: float, handle_minutes: float, target_seconds: float, target: float) -> int:
    """Smallest number of advisors that reaches the service level target."""
    if calls_per_hour <= 0:
        return 0
    traffic = calls_per_hour * handle_minutes / 60
    advisors = max(1, math.ceil(traffic))
    while service_level(advisors, traffic, handle_minutes, target_seconds) < target:
        advisors += 1
    return advisors


# ---------------------------------------------------------------- the baseline

class FixedRulePolicy(Policy):
    """Staffing plan decided in advance from the forecast, the same every day."""

    def __init__(self, config: CenterConfig | None = None):
        self.config = config or CenterConfig()
        cfg = self.config
        self.plan = []                               # one {queue: advisors} per opening hour
        for calls in cfg.calls_per_hour:
            self.plan.append({
                q: advisors_needed(calls * cfg.call_mix[q], cfg.mean_handle_minutes[q], cfg.sla_seconds, cfg.sla_target)
                for q in QUEUES
            })

    def decide(self, snapshot: Snapshot) -> dict:
        hour = min(int(snapshot.minute // 60), len(self.plan) - 1)
        return dict(self.plan[hour])


class FixedStaffPolicy(Policy):
    """Even simpler reference: the same team all day, whatever the hour."""

    def __init__(self, staff: dict):
        self.staff = staff

    def decide(self, snapshot: Snapshot) -> dict:
        return dict(self.staff)
