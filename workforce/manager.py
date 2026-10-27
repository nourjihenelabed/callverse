"""Floor manager: learns where to place workers by practicing many pretend days.

This file has the policies that decide how many advisors work on each queue:
- FixedRulePolicy: the classic baseline. A staffing plan computed in advance,
  hour by hour, with the Erlang C formula that real call centers use. It never
  reacts to what happens during the day.
- MarginRulePolicy: the same plan, plus a fixed safety margin of advisors, priority on.
- ReactiveRulePolicy: the same plan, plus extra advisors when a queue builds up, and
  multi-skilled advisors serve specialist queues first when customers wait there.
- The RL agent lives in env.py. All policies are compared in compare.py.
The two rules' settings are tuned for the calibrated center by workforce.tune_rules.
"""

import json
import math
from pathlib import Path

from workforce.simulation import QUEUES, CenterConfig, Policy, Snapshot

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

# Used when calibration.json has no "rules" section (python -m workforce.tune_rules writes one).
DEFAULT_RULES = {
    "reactive": {"per_waiting": 1, "max_extra": 5, "use_priority": False},
    "margin": {"margin": {"general": 4, "technical": 0, "complaint": 0}, "priority": True},
}


def tuned_rules() -> dict:
    """The rule settings tuned for the calibrated center, or the defaults."""
    try:
        rules = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")).get("rules", {})
    except FileNotFoundError:
        rules = {}
    return {name: {**DEFAULT_RULES[name], **{k: v for k, v in rules.get(name, {}).items() if k in DEFAULT_RULES[name]}}
            for name in DEFAULT_RULES}


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

    # settings tuned on random training days (never the test days) by workforce.tune_rules
    def __init__(self, config: CenterConfig | None = None, per_waiting: int | None = None,
                 max_extra: int | None = None, use_priority: bool | None = None):
        super().__init__(config)
        tuned = tuned_rules()["reactive"]
        self.per_waiting = per_waiting if per_waiting is not None else tuned["per_waiting"]
        self.max_extra = max_extra if max_extra is not None else tuned["max_extra"]
        self.use_priority = use_priority if use_priority is not None else tuned["use_priority"]

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

    def __init__(self, config: CenterConfig | None = None, margin: dict | None = None,
                 priority: bool | None = None):
        super().__init__(config)
        tuned = tuned_rules()["margin"]
        self.margin = margin or dict(tuned["margin"])
        self.priority = priority if priority is not None else tuned["priority"]

    def decide(self, snapshot: Snapshot) -> dict:
        plan = super().decide(snapshot)
        return {q: plan[q] + self.margin[q] for q in QUEUES} | {"priority": self.priority}


class FixedStaffPolicy(Policy):
    """Even simpler reference: the same team all day, whatever the hour."""

    def __init__(self, staff: dict):
        self.staff = staff

    def decide(self, snapshot: Snapshot) -> dict:
        return dict(self.staff)
