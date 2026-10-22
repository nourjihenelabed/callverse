"""Floor manager: learns where to place workers by practicing many pretend days.

This file has the policies that decide how many advisors work on each queue:
- FixedRulePolicy: the classic baseline. A staffing plan computed in advance,
  hour by hour, with the Erlang C formula that real call centers use. It never
  reacts to what happens during the day.
- MarginRulePolicy: the same plan, plus a fixed safety margin of advisors, priority on.
- ReactiveRulePolicy: the same plan, plus extra advisors when a queue builds up, and
  multi-skilled advisors serve specialist queues first when customers wait there.
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


class ReactiveRulePolicy(FixedRulePolicy):
    """A supervisor's simple reflex: follow the Erlang C plan, and call in extra
    advisors when customers pile up in a queue. A fairer opponent for the RL agent,
    because it also reacts to what happens during the day."""

    # tuned on training days (not the test days): 1 extra advisor per waiting customer, at most 5;
    # turning priority on made this rule slightly worse, so it is off by default
    def __init__(self, config: CenterConfig | None = None, per_waiting: int = 1, max_extra: int = 5,
                 use_priority: bool = False):
        super().__init__(config)
        self.per_waiting, self.max_extra, self.use_priority = per_waiting, max_extra, use_priority

    def decide(self, snapshot: Snapshot) -> dict:
        plan = super().decide(snapshot)
        wanted = {q: plan[q] + min(self.max_extra, snapshot.waiting[q] // self.per_waiting) for q in QUEUES}
        # specialists on general calls help when technical or complaint customers are waiting
        wanted["priority"] = self.use_priority and (snapshot.waiting["technical"] + snapshot.waiting["complaint"]) > 0
        return wanted


class MarginRulePolicy(FixedRulePolicy):
    """The Erlang C plan plus a fixed safety margin per queue, with priority on.
    A strong, simple opponent: it was found by looking at what the first RL agent
    actually did (see README). Margin tuned on random training days."""

    def __init__(self, config: CenterConfig | None = None, margin: dict | None = None, priority: bool = True):
        super().__init__(config)
        self.margin = margin or {"general": 4, "technical": 0, "complaint": 0}
        self.priority = priority

    def decide(self, snapshot: Snapshot) -> dict:
        plan = super().decide(snapshot)
        return {q: plan[q] + self.margin[q] for q in QUEUES} | {"priority": self.priority}


class FixedStaffPolicy(Policy):
    """Even simpler reference: the same team all day, whatever the hour."""

    def __init__(self, staff: dict):
        self.staff = staff

    def decide(self, snapshot: Snapshot) -> dict:
        return dict(self.staff)
