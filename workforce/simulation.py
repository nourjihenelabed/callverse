"""The pretend call center: sends in customers, keeps the waiting lines, runs the clock.

One simulated day:
- Calls arrive all day, more at busy hours (10h and 14h), in three queues:
  general questions, technical problems, complaints.
- Advisors have skills. Each one works on one queue at a time, or is off duty.
- A customer who waits longer than their patience hangs up (abandonment).
- Every 15 simulated minutes, a policy (the floor manager) looks at the center
  and decides how many advisors work on each queue. A policy is either the
  fixed-rule baseline or the RL agent: they answer the same question.

No AI model and no database here: call durations and patience come from
statistics, so thousands of days can be simulated quickly for RL training.
The statistical defaults in CenterConfig are calibrated from published
Technion bank call-center results. The JSON file records their sources.

Try it:  python -m workforce.simulation
"""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import simpy

QUEUES = ("general", "technical", "complaint")


# ---------------------------------------------------------------- surprise peaks

@dataclass(frozen=True)
class Surge:
    """Calls the forecast does not know about: `factor` times the usual calls of
    `queue` between start_hour and end_hour (clock hours, e.g. 10 to 13).
    A factor below 1 means fewer calls than forecast (a quiet day)."""

    queue: str
    start_hour: int
    end_hour: int
    factor: float


SCENARIOS = {
    "normal": (),
    # the network goes down: technical calls double, angry customers call back with complaints
    "network_outage": (Surge("technical", 10, 13, 2.0), Surge("complaint", 11, 14, 1.5)),
    # bills are sent: more questions and complaints all day
    "billing_day": (Surge("general", 8, 20, 1.3), Surge("complaint", 8, 20, 1.6)),
    # an advertising campaign: many people call about the offer in the afternoon
    "afternoon_promo": (Surge("general", 14, 17, 1.8),),
}

# Never used in training: they test whether the agent learned to react, or only memorized
# the training peaks.
HELD_OUT_SCENARIOS = {
    # the same outage, but in the late afternoon instead of the morning
    "late_outage": (Surge("technical", 16, 19, 2.0), Surge("complaint", 17, 20, 1.5)),
    # a bigger outage than any seen in training
    "big_outage": (Surge("technical", 10, 13, 2.5), Surge("complaint", 11, 14, 1.8)),
    # complaints pile up on their own (in training they only rose with other queues)
    "complaint_storm": (Surge("complaint", 9, 12, 2.5),),
    # 30% fewer calls than forecast all day: does the agent waste money on staff?
    "quiet_day": tuple(Surge(q, 8, 20, 0.7) for q in QUEUES),
}

ALL_SCENARIOS = SCENARIOS | HELD_OUT_SCENARIOS


# ---------------------------------------------------------------- settings of the pretend center

@dataclass
class CenterConfig:
    open_hour: int = 8
    close_hour: int = 20
    # calls per hour, all queues together, from open_hour to close_hour
    calls_per_hour: tuple = (60, 90, 120, 110, 85, 80, 105, 115, 95, 80, 65, 45)
    call_mix: dict = field(default_factory=lambda: {"general": 0.55, "technical": 0.30, "complaint": 0.15})
    mean_handle_minutes: dict | None = None
    handle_spread: float | None = None         # log-normal spread of call durations
    handle_spread_by_queue: dict | None = None
    mean_patience_minutes: dict | None = None
    forecast_error: float | None = None        # maximum +/- error per 15-minute period
    # the advisors on the roster: (how many, which queues they can work on)
    # The calibrated roster is resized to keep the same staffing proportion.
    roster: tuple | None = None
    decision_minutes: int = 15                 # how often the policy decides
    min_shift_minutes: int = 60                # an advisor called in stays on duty at least this long
    switch_minutes: float = 2.0                # time to set up when moving to another queue (paid, no calls)
    sla_seconds: int = 20                      # target: answer within 20 seconds...
    sla_target: float = 0.80                   # ...for 80% of calls
    cost_per_advisor_hour: float = 25.0        # euros, for each advisor on duty
    overtime_minutes: int = 60                 # advisors finish waiting calls after closing

    def __post_init__(self):
        """Load Phase 1 values while retaining explicit custom values and fallbacks."""
        defaults = {
            "mean_handle_minutes": {"general": 5.0, "technical": 9.0, "complaint": 12.0},
            "handle_spread": 0.6,
            "handle_spread_by_queue": {"general": 0.6, "technical": 0.6, "complaint": 0.6},
            "mean_patience_minutes": {"general": 3.0, "technical": 4.0, "complaint": 5.0},
            "forecast_error": 0.0,
            "roster": ((9, ("general",)), (12, ("general", "technical")),
                       (9, ("general", "complaint"))),
        }
        path = Path(__file__).with_name("calibration.json")
        if path.exists():
            with path.open(encoding="utf-8") as file:
                calibration = json.load(file)
            service = calibration["service"]
            explicit_spread = self.handle_spread
            self.mean_handle_minutes = self.mean_handle_minutes or {
                queue: service[queue]["mean_minutes"] for queue in QUEUES
            }
            self.handle_spread = self.handle_spread if self.handle_spread is not None else service["general"]["spread"]
            self.handle_spread_by_queue = self.handle_spread_by_queue or {
                queue: explicit_spread if explicit_spread is not None else service[queue]["spread"]
                for queue in QUEUES
            }
            self.mean_patience_minutes = self.mean_patience_minutes or dict(calibration["patience"]["minutes"])
            forecast_error = calibration["forecast_error"]
            self.forecast_error = (self.forecast_error if self.forecast_error is not None
                                   else forecast_error["maximum"])
            roster = calibration["roster"]
            self.roster = self.roster or (
                (roster["general_only"], ("general",)),
                (roster["general_technical"], ("general", "technical")),
                (roster["general_complaint"], ("general", "complaint")),
            )
        else:
            for name, value in defaults.items():
                if getattr(self, name) is None:
                    setattr(self, name, value.copy() if isinstance(value, dict) else value)

    @property
    def day_minutes(self) -> int:
        return (self.close_hour - self.open_hour) * 60


# ---------------------------------------------------------------- what the policy sees and decides

@dataclass
class Snapshot:
    """What the floor manager sees at each decision. The RL agent's observation comes from this."""

    minute: float                              # minutes since opening
    hour: int                                  # clock hour, e.g. 10
    waiting: dict                              # customers waiting, per queue
    staffed: dict                              # advisors working, per queue
    on_duty: int                               # advisors on duty in total
    roster_size: int
    last_avg_wait_s: dict                      # average wait of calls answered in the last period, per queue
    last_sla: float                            # share of last period's calls answered within the SLA
    last_abandoned: int                        # customers who hung up in the last period
    calls_next_hour: float                     # forecast of calls in the coming hour
    priority: bool = False                     # are multi-skilled advisors serving specialist queues first?
    last_calls: dict = field(default_factory=dict)      # calls received in the last period, per queue
    expected_calls: dict = field(default_factory=dict)  # calls the forecast expected in that period, per queue


class Policy:
    """A floor manager. decide() returns how many advisors work on each queue.

    The answer is a dict like {"general": 8, "technical": 5, "complaint": 3}.
    Advisors not asked for are off duty (and cost nothing). If a queue asks for
    more advisors than have that skill, it gets as many as possible.
    Realistic limits apply: an advisor called in stays at least min_shift_minutes,
    and moving to another queue takes switch_minutes.
    Add "priority": True to let multi-skilled advisors working on general calls
    serve waiting technical or complaint calls first.
    """

    def decide(self, snapshot: Snapshot) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------- results

@dataclass
class DayResult:
    offered: dict
    answered: dict
    abandoned: dict
    within_sla: dict
    wait_seconds_total: dict
    satisfaction_total: float
    busy_minutes: float
    on_duty_minutes: float
    staff_cost: float
    unanswered_at_close: int
    switches: int = 0                          # times an advisor moved to another queue

    def kpis(self) -> dict:
        offered = sum(self.offered.values())
        answered = sum(self.answered.values())
        return {
            "calls": offered,
            "answered": answered,
            "abandonment_rate": round(sum(self.abandoned.values()) / offered, 4) if offered else 0.0,
            "sla": round(sum(self.within_sla.values()) / offered, 4) if offered else 0.0,
            "avg_wait_s": round(sum(self.wait_seconds_total.values()) / answered, 1) if answered else 0.0,
            "occupancy": round(self.busy_minutes / self.on_duty_minutes, 4) if self.on_duty_minutes else 0.0,
            "staff_cost_eur": round(self.staff_cost, 2),
            "satisfaction": round(self.satisfaction_total / offered, 3) if offered else 0.0,
            "queue_switches": self.switches,
            "per_queue": {
                q: {
                    "calls": self.offered[q],
                    "abandonment_rate": round(self.abandoned[q] / self.offered[q], 4) if self.offered[q] else 0.0,
                    "sla": round(self.within_sla[q] / self.offered[q], 4) if self.offered[q] else 0.0,
                    "avg_wait_s": round(self.wait_seconds_total[q] / self.answered[q], 1) if self.answered[q] else 0.0,
                }
                for q in QUEUES
            },
        }


def satisfaction_after(wait_minutes: float | None) -> float:
    """PLACEHOLDER score from 1 to 5 until it comes from real LangGraph calls.
    None means the customer hung up."""
    if wait_minutes is None:
        return 1.0
    return max(1.0, 4.5 - 0.5 * wait_minutes)


# ---------------------------------------------------------------- the simulation

class _Call:
    __slots__ = ("queue", "arrived", "picked", "abandoned")

    def __init__(self, env, queue):
        self.queue, self.arrived = queue, env.now
        self.picked, self.abandoned = env.event(), False


class CallCenter:
    """One simulated day. Usage: CallCenter(policy, seed=1).run().kpis()"""

    def __init__(self, policy: Policy, seed: int = 0, config: CenterConfig | None = None, scenario="normal"):
        """`scenario` is a name from ALL_SCENARIOS, or a tuple of Surge (used for random training days)."""
        self.policy, self.config = policy, config or CenterConfig()
        self.scenario = scenario
        self.surges = ALL_SCENARIOS[scenario] if isinstance(scenario, str) else tuple(scenario)
        self._drop_rng = random.Random(seed * 1000 + 999)   # own random numbers for quiet periods
        self.seed = seed
        self.rng = random.Random(seed)
        forecast_rng = random.Random(seed * 1000 + 997)
        periods = math.ceil(self.config.day_minutes / self.config.decision_minutes)
        error = self.config.forecast_error
        self._forecast_factors = [
            forecast_rng.uniform(1 - error, 1 + error) for _ in range(periods)
        ]
        self.env = simpy.Environment()
        self.queues = {q: simpy.Store(self.env) for q in QUEUES}
        self.advisors = [skills for count, skills in self.config.roster for _ in range(count)]
        self.assignment = [None] * len(self.advisors)    # queue name, or None when off duty
        self.busy = [False] * len(self.advisors)
        self.on_since = [None] * len(self.advisors)       # when each advisor was called in
        self.serving = [None] * len(self.advisors)        # the queue each advisor is set up for
        self.priority = False
        self.waiting = {q: 0 for q in QUEUES}
        zeros = lambda: {q: 0 for q in QUEUES}  # noqa: E731
        self.result = DayResult(zeros(), zeros(), zeros(), zeros(), {q: 0.0 for q in QUEUES},
                                0.0, 0.0, 0.0, 0.0, 0)
        self._period = {"waits": {q: [] for q in QUEUES}, "answered": 0, "within": 0, "abandoned": 0, "offered": 0,
                        "offered_q": {q: 0 for q in QUEUES}, "start": 0.0}
        self._wakeups = [self.env.event() for _ in self.advisors]

    # ---- the day
    @property
    def end_minute(self) -> int:
        return self.config.day_minutes + self.config.overtime_minutes

    def run(self) -> DayResult:
        """Runs the whole day at once."""
        self.run_until(self.end_minute)
        return self.result

    def run_until(self, minute: float) -> None:
        """Runs the day up to `minute` (used by the RL environment, one decision at a time)."""
        if not getattr(self, "_started", False):
            self._started = True
            self.env.process(self._manager())
            self.env.process(self._arrivals())
            for k, surge in enumerate(self.surges):
                self.env.process(self._surge_arrivals(surge, random.Random(self.seed * 1000 + k + 1)))
            for i in range(len(self.advisors)):
                self.env.process(self._advisor(i))
        self.env.run(until=min(minute, self.end_minute))
        self.result.unanswered_at_close = sum(self.waiting.values())

    @property
    def finished(self) -> bool:
        return self.env.now >= self.end_minute

    # ---- customers
    def _rate_at(self, minute: float) -> float:
        hour = int(minute // 60)
        rates = self.config.calls_per_hour
        return rates[hour] / 60 if 0 <= hour < len(rates) else 0.0

    def _arrivals(self):
        cfg = self.config
        peak = max(self.config.calls_per_hour) * (1 + cfg.forecast_error) / 60
        while True:                                 # thinning: arrival rate changes with the hour
            yield self.env.timeout(self.rng.expovariate(peak))
            if self.env.now >= cfg.day_minutes:
                return
            period = min(int(self.env.now // cfg.decision_minutes), len(self._forecast_factors) - 1)
            rate = self._rate_at(self.env.now) * self._forecast_factors[period]
            if self.rng.random() < rate / peak:
                queue = self.rng.choices(QUEUES, weights=[cfg.call_mix[q] for q in QUEUES])[0]
                if self._dropped(queue):            # quieter than forecast: this call never comes
                    continue
                self.env.process(self._customer(queue))

    def _dropped(self, queue) -> bool:
        clock = self.config.open_hour + self.env.now / 60
        for surge in self.surges:
            if surge.factor < 1 and surge.queue == queue and surge.start_hour <= clock < surge.end_hour:
                return self._drop_rng.random() >= surge.factor
        return False

    def _surge_arrivals(self, surge: Surge, rng: random.Random):
        if surge.factor <= 1:                       # quiet periods are handled in _arrivals
            return
            yield
        """The extra calls of a surprise peak, on top of the usual ones (own random numbers,
        so a normal day stays exactly the same)."""
        cfg = self.config
        start = (surge.start_hour - cfg.open_hour) * 60
        end = min((surge.end_hour - cfg.open_hour) * 60, cfg.day_minutes)
        extra = lambda minute: self._rate_at(minute) * cfg.call_mix[surge.queue] * (surge.factor - 1)  # noqa: E731
        peak = max(extra(start + 60 * h) for h in range(max(1, (end - start) // 60))) or 1e-9
        yield self.env.timeout(max(0, start))
        while True:
            yield self.env.timeout(rng.expovariate(peak))
            if self.env.now >= end:
                return
            if rng.random() < extra(self.env.now) / peak:
                self.env.process(self._customer(surge.queue))

    def _customer(self, queue):
        cfg, call = self.config, _Call(self.env, queue)
        self.result.offered[queue] += 1
        self._period["offered"] += 1
        self._period["offered_q"][queue] += 1
        self.waiting[queue] += 1
        yield self.queues[queue].put(call)
        self._wake(queue)
        patience = self.rng.expovariate(1 / cfg.mean_patience_minutes[queue])
        outcome = yield call.picked | self.env.timeout(patience)
        if call.picked not in outcome:              # hung up before an advisor took the call
            call.abandoned = True
            self.waiting[queue] -= 1
            self.result.abandoned[queue] += 1
            self._period["abandoned"] += 1
            self.result.satisfaction_total += satisfaction_after(None)

    # ---- advisors
    def _can_take(self, i, queue) -> bool:
        assigned = self.assignment[i]
        return assigned == queue or (self.priority and assigned == "general" and queue in self.advisors[i])

    def _wake(self, queue):
        for i in range(len(self.advisors)):
            if self._can_take(i, queue) and not self.busy[i] and not self._wakeups[i].triggered:
                self._wakeups[i].succeed()

    def _advisor(self, i):
        cfg = self.config
        while True:
            queue = self.assignment[i]
            if queue and self.serving[i] not in (None, queue):     # moved to another queue: set up first
                self.result.switches += 1
                self.serving[i] = queue
                yield self.env.timeout(cfg.switch_minutes)
                continue
            if queue:
                self.serving[i] = queue
            call = self._next_call_for(i, queue) if queue else None
            if call is None:                        # nothing to do: wait for a call or a new assignment
                self._wakeups[i] = self.env.event()
                yield self._wakeups[i]
                continue
            self.busy[i] = True
            call.picked.succeed()
            queue = call.queue                      # with priority, may differ from the assigned queue
            self.waiting[queue] -= 1
            wait = self.env.now - call.arrived
            self.result.answered[queue] += 1
            self.result.wait_seconds_total[queue] += wait * 60
            if wait * 60 <= cfg.sla_seconds:
                self.result.within_sla[queue] += 1
                self._period["within"] += 1
            self._period["waits"][queue].append(wait * 60)
            self._period["answered"] += 1
            self.result.satisfaction_total += satisfaction_after(wait)
            mean = cfg.mean_handle_minutes[queue]
            sigma = cfg.handle_spread_by_queue[queue]
            duration = self.rng.lognormvariate(math.log(mean) - sigma ** 2 / 2, sigma)
            yield self.env.timeout(duration)
            self.result.busy_minutes += duration
            self.busy[i] = False

    def _next_call_for(self, i, queue):
        if self.priority and queue == "general":
            for other in self.advisors[i]:
                if other != "general":
                    call = self._next_call(other)
                    if call is not None:
                        return call
        return self._next_call(queue)

    def _next_call(self, queue):
        items = self.queues[queue].items
        while items:
            call = items.pop(0)
            if not call.abandoned:
                return call
        return None

    # ---- the floor manager
    def _snapshot(self) -> Snapshot:
        now, cfg, p = self.env.now, self.config, self._period
        staffed = {q: sum(1 for a in self.assignment if a == q) for q in QUEUES}
        hour = int(now // 60)
        forecast = cfg.calls_per_hour[hour] if hour < len(cfg.calls_per_hour) else 0
        return Snapshot(
            minute=now,
            hour=cfg.open_hour + hour,
            waiting=dict(self.waiting),
            staffed=staffed,
            on_duty=sum(staffed.values()),
            roster_size=len(self.advisors),
            last_avg_wait_s={q: (sum(w) / len(w) if w else 0.0) for q, w in p["waits"].items()},
            last_sla=p["within"] / p["offered"] if p["offered"] else 1.0,
            last_abandoned=p["abandoned"],
            calls_next_hour=float(forecast),
            priority=self.priority,
            last_calls=dict(p["offered_q"]),
            expected_calls=self._expected_calls(p["start"], now),
        )

    def _expected_calls(self, start: float, end: float) -> dict:
        """Calls the forecast expected between two moments, per queue (whole minutes)."""
        expected = sum(self._rate_at(m + 0.5) for m in range(int(start), int(end)))
        return {q: expected * self.config.call_mix[q] for q in QUEUES}

    def _apply(self, wanted: dict, keep_min_shift: bool = True):
        """Gives each queue the number of advisors asked for, using skills.
        Busy advisors finish their call first, then switch. Advisors called in less
        than min_shift_minutes ago stay on duty even if the policy asks for fewer."""
        new = [None] * len(self.advisors)
        free = list(range(len(self.advisors)))
        # rarest skills first, so specialists are not used up on general calls
        for queue in sorted(QUEUES, key=lambda q: sum(q in s for s in self.advisors)):
            need = max(0, int(wanted.get(queue, 0)))
            # prefer advisors already on this queue, then the least versatile ones
            able = sorted((i for i in free if queue in self.advisors[i]),
                          key=lambda i: (self.assignment[i] != queue, len(self.advisors[i])))
            for i in able[:need]:
                new[i] = queue
                free.remove(i)
        now = self.env.now
        if keep_min_shift:
            for i in free:
                since = self.on_since[i]
                if self.assignment[i] and since is not None and now - since < self.config.min_shift_minutes:
                    new[i] = self.assignment[i]      # too early to send this advisor home
        for i in range(len(new)):
            if new[i] and not self.assignment[i]:
                self.on_since[i] = now
            elif not new[i]:
                self.on_since[i] = None
                self.serving[i] = None
        self.priority = bool(wanted.get("priority", False))
        self.assignment = new
        for i, queue in enumerate(new):
            if queue and not self.busy[i] and not self._wakeups[i].triggered:
                self._wakeups[i].succeed()

    def _manager(self):
        cfg = self.config
        while self.env.now < self.end_minute:
            closed = self.env.now >= cfg.day_minutes
            wanted = self.policy.decide(self._snapshot())
            if closed:                              # after closing: only finish waiting calls
                wanted = {q: (wanted.get(q, 0) if self.waiting[q] else 0) for q in QUEUES} | {
                    "priority": wanted.get("priority", False)}
            self._apply(wanted, keep_min_shift=not closed)
            self._period = {"waits": {q: [] for q in QUEUES}, "answered": 0, "within": 0, "abandoned": 0, "offered": 0,
                        "offered_q": {q: 0 for q in QUEUES}, "start": self.env.now}
            # paid for the coming period: advisors assigned to a queue, plus those still finishing a call
            on_duty = sum(1 for a, b in zip(self.assignment, self.busy) if a or b)
            minutes = min(cfg.decision_minutes, self.end_minute - self.env.now)
            self.result.on_duty_minutes += on_duty * minutes
            self.result.staff_cost += on_duty * minutes / 60 * cfg.cost_per_advisor_hour
            yield self.env.timeout(cfg.decision_minutes)


def simulate_days(policy: Policy, seeds, config: CenterConfig | None = None, scenario: str = "normal") -> dict:
    """Runs several days and averages the main KPIs (used to compare policies fairly)."""
    keys = ("abandonment_rate", "sla", "avg_wait_s", "occupancy", "staff_cost_eur", "satisfaction")
    runs = [CallCenter(policy, seed=s, config=config, scenario=scenario).run().kpis() for s in seeds]
    return {k: round(sum(r[k] for r in runs) / len(runs), 4) for k in keys} | {"days": len(runs)}


if __name__ == "__main__":
    from workforce.manager import FixedRulePolicy

    day = CallCenter(FixedRulePolicy(), seed=1).run().kpis()
    print("One day with the fixed-rule baseline:")
    for key, value in day.items():
        if key != "per_queue":
            print(f"  {key:<17} {value}")
    print("  per queue:")
    for queue, values in day["per_queue"].items():
        print(f"    {queue:<10} {values}")
    print("\nAverage over 20 days:", simulate_days(FixedRulePolicy(), range(20)))
