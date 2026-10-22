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
The numbers in CenterConfig are PLACEHOLDERS, to calibrate later on public
call center data (Technion) and on statistics from real LangGraph calls.

Try it:  python -m workforce.simulation
"""

import math
import random
from dataclasses import dataclass, field

import simpy

QUEUES = ("general", "technical", "complaint")


# ---------------------------------------------------------------- settings of the pretend center

@dataclass
class CenterConfig:
    open_hour: int = 8
    close_hour: int = 20
    # calls per hour, all queues together, from open_hour to close_hour
    calls_per_hour: tuple = (60, 90, 120, 110, 85, 80, 105, 115, 95, 80, 65, 45)
    call_mix: dict = field(default_factory=lambda: {"general": 0.55, "technical": 0.30, "complaint": 0.15})
    mean_handle_minutes: dict = field(default_factory=lambda: {"general": 5.0, "technical": 9.0, "complaint": 12.0})
    handle_spread: float = 0.6                 # log-normal spread of call durations
    mean_patience_minutes: dict = field(default_factory=lambda: {"general": 3.0, "technical": 4.0, "complaint": 5.0})
    # the advisors on the roster: (how many, which queues they can work on)
    roster: tuple = (
        (9, ("general",)),
        (9, ("general", "technical")),
        (6, ("general", "complaint")),
    )
    decision_minutes: int = 15                 # how often the policy decides
    sla_seconds: int = 20                      # target: answer within 20 seconds...
    sla_target: float = 0.80                   # ...for 80% of calls
    cost_per_advisor_hour: float = 25.0        # euros, for each advisor on duty
    overtime_minutes: int = 60                 # advisors finish waiting calls after closing

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


class Policy:
    """A floor manager. decide() returns how many advisors work on each queue.

    The answer is a dict like {"general": 8, "technical": 5, "complaint": 3}.
    Advisors not asked for are off duty (and cost nothing). If a queue asks for
    more advisors than have that skill, it gets as many as possible.
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

    def __init__(self, policy: Policy, seed: int = 0, config: CenterConfig | None = None):
        self.policy, self.config = policy, config or CenterConfig()
        self.rng = random.Random(seed)
        self.env = simpy.Environment()
        self.queues = {q: simpy.Store(self.env) for q in QUEUES}
        self.advisors = [skills for count, skills in self.config.roster for _ in range(count)]
        self.assignment = [None] * len(self.advisors)    # queue name, or None when off duty
        self.busy = [False] * len(self.advisors)
        self.waiting = {q: 0 for q in QUEUES}
        zeros = lambda: {q: 0 for q in QUEUES}  # noqa: E731
        self.result = DayResult(zeros(), zeros(), zeros(), zeros(), {q: 0.0 for q in QUEUES},
                                0.0, 0.0, 0.0, 0.0, 0)
        self._period = {"waits": {q: [] for q in QUEUES}, "answered": 0, "within": 0, "abandoned": 0, "offered": 0}
        self._wakeups = [self.env.event() for _ in self.advisors]

    # ---- the day
    def run(self) -> DayResult:
        cfg = self.config
        self.env.process(self._manager())
        self.env.process(self._arrivals())
        for i in range(len(self.advisors)):
            self.env.process(self._advisor(i))
        self.env.run(until=cfg.day_minutes + cfg.overtime_minutes)
        self.result.unanswered_at_close = sum(self.waiting.values())
        return self.result

    # ---- customers
    def _rate_at(self, minute: float) -> float:
        hour = int(minute // 60)
        rates = self.config.calls_per_hour
        return rates[hour] / 60 if 0 <= hour < len(rates) else 0.0

    def _arrivals(self):
        cfg, peak = self.config, max(self.config.calls_per_hour) / 60
        while True:                                 # thinning: arrival rate changes with the hour
            yield self.env.timeout(self.rng.expovariate(peak))
            if self.env.now >= cfg.day_minutes:
                return
            if self.rng.random() < self._rate_at(self.env.now) / peak:
                queue = self.rng.choices(QUEUES, weights=[cfg.call_mix[q] for q in QUEUES])[0]
                self.env.process(self._customer(queue))

    def _customer(self, queue):
        cfg, call = self.config, _Call(self.env, queue)
        self.result.offered[queue] += 1
        self._period["offered"] += 1
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
    def _wake(self, queue):
        for i, assigned in enumerate(self.assignment):
            if assigned == queue and not self.busy[i] and not self._wakeups[i].triggered:
                self._wakeups[i].succeed()

    def _advisor(self, i):
        cfg = self.config
        while True:
            queue = self.assignment[i]
            call = self._next_call(queue) if queue else None
            if call is None:                        # nothing to do: wait for a call or a new assignment
                self._wakeups[i] = self.env.event()
                yield self._wakeups[i]
                continue
            self.busy[i] = True
            call.picked.succeed()
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
            sigma = cfg.handle_spread
            duration = self.rng.lognormvariate(math.log(mean) - sigma ** 2 / 2, sigma)
            yield self.env.timeout(duration)
            self.result.busy_minutes += duration
            self.busy[i] = False

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
        )

    def _apply(self, wanted: dict):
        """Gives each queue the number of advisors asked for, using skills.
        Busy advisors finish their call first, then switch."""
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
        self.assignment = new
        for i, queue in enumerate(new):
            if queue and not self.busy[i] and not self._wakeups[i].triggered:
                self._wakeups[i].succeed()

    def _manager(self):
        cfg = self.config
        while self.env.now < cfg.day_minutes + cfg.overtime_minutes:
            closed = self.env.now >= cfg.day_minutes
            wanted = self.policy.decide(self._snapshot())
            if closed:                              # after closing: only finish waiting calls
                wanted = {q: (wanted.get(q, 0) if self.waiting[q] else 0) for q in QUEUES}
            self._apply(wanted)
            self._period = {"waits": {q: [] for q in QUEUES}, "answered": 0, "within": 0, "abandoned": 0, "offered": 0}
            # paid: advisors assigned to a queue, plus those still finishing a call
            on_duty = sum(1 for a, b in zip(self.assignment, self.busy) if a or b)
            start = self.env.now
            yield self.env.timeout(cfg.decision_minutes)
            self.result.on_duty_minutes += on_duty * (self.env.now - start)
            self.result.staff_cost += on_duty * (self.env.now - start) / 60 * cfg.cost_per_advisor_hour


def simulate_days(policy: Policy, seeds, config: CenterConfig | None = None) -> dict:
    """Runs several days and averages the main KPIs (used to compare policies fairly)."""
    keys = ("abandonment_rate", "sla", "avg_wait_s", "occupancy", "staff_cost_eur", "satisfaction")
    runs = [CallCenter(policy, seed=s, config=config).run().kpis() for s in seeds]
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
