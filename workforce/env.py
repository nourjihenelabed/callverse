"""The simulated call center as a Gymnasium environment, so an RL agent can practice on it.

- One episode = one day. One step = one decision, every 15 simulated minutes (52 per day).
  Training days are random: normal, quieter than forecast, or random surprise peaks
  (random_surprises). The agent is not told which. The named scenarios are kept for testing.
- Observation: what the floor manager sees (time, queues, staffing, last period's results, forecast).
- Action: a correction to the Erlang C plan, for each queue: -3 to +3 advisors, plus
  "priority" on or off (multi-skilled advisors on general calls serve specialist queues first).
  Starting from the classic plan makes learning much faster and keeps the agent
  reasonable from the first day; the agent learns when to add or remove advisors.
- Reward: minus the cost of the period in euros, counting staff pay AND a price
  for bad service (a customer who hangs up, a call answered after the 20 s target).
  These prices are a team choice: see RewardConfig.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from workforce.manager import FixedRulePolicy
from workforce.simulation import QUEUES, SCENARIOS, CallCenter, CenterConfig, Policy, Snapshot, Surge

MAX_CHANGE = 3                                  # the agent moves the plan by -3..+3 advisors per queue

def random_surprises(rng: np.random.Generator) -> tuple:
    """A random training day: normal, quieter than forecast, or 1-2 surprise peaks of
    random queue, time, length and size. The agent is never told which: it has to
    notice from the calls coming in. The named test scenarios are never used in training."""
    kind = rng.random()
    if kind < 0.25:
        return ()                                                       # a normal day
    if kind < 0.40:                                                     # quieter than forecast
        factor = float(rng.uniform(0.6, 0.9))
        start = int(rng.integers(8, 15))
        return tuple(Surge(q, start, min(20, start + int(rng.integers(3, 13))), factor) for q in QUEUES)
    surges = []
    for _ in range(int(rng.integers(1, 3))):                            # 1 or 2 surprise peaks
        queue = QUEUES[int(rng.integers(len(QUEUES)))]
        start = int(rng.integers(8, 18))
        surges.append(Surge(queue, start, min(20, start + int(rng.integers(2, 6))), float(rng.uniform(1.3, 2.6))))
    return tuple(surges)


@dataclass
class RewardConfig:
    """How much each problem costs, in euros. A team choice: see workforce/README.md.

    With cheap prices (8 € / 2 €) the agent learned to cut staff and let service
    collapse (SLA 71%, 13% hang-ups). These values make bad service cost what it
    costs a telecom operator (callbacks, unhappy customers, churn)."""

    abandoned_call: float = 50.0                # a customer who hangs up: may call back, complain or leave
    late_answer: float = 8.0                    # a call answered after the SLA target
    scale: float = 100.0                        # rewards are divided by this to help training


def observation(snapshot: Snapshot, plan: FixedRulePolicy, config: CenterConfig) -> np.ndarray:
    """Turns a Snapshot into the numbers the agent sees (all roughly between 0 and 1)."""
    hour = min(int(snapshot.minute // 60), len(plan.plan) - 1)
    end = config.day_minutes + config.overtime_minutes
    values = [snapshot.minute / end]
    for q in QUEUES:
        values += [
            snapshot.waiting[q] / 30,
            snapshot.staffed[q] / 10,
            snapshot.last_avg_wait_s[q] / 60,
            plan.plan[hour][q] / 10,
        ]
    values += [snapshot.last_sla, snapshot.last_abandoned / 10, snapshot.calls_next_hour / 120,
               float(snapshot.priority)]
    # calls received in the last 15 minutes compared with the forecast: 0.5 = as expected, higher = surge
    for q in QUEUES:
        values.append(snapshot.last_calls.get(q, 0) / (snapshot.expected_calls.get(q, 0) + 1) / 2)
    return np.clip(np.array(values, dtype=np.float32), 0.0, 5.0)


def apply_correction(snapshot: Snapshot, plan: FixedRulePolicy, action) -> dict:
    """Erlang C plan for this hour, corrected by the agent's action."""
    planned = plan.decide(snapshot)
    wanted = {q: max(0, planned[q] + int(action[i]) - MAX_CHANGE) for i, q in enumerate(QUEUES)}
    wanted["priority"] = bool(action[len(QUEUES)])
    return wanted


class _Waiting(Policy):
    """Inside the environment, the decision comes from step(), not from a rule."""

    def __init__(self, plan: FixedRulePolicy):
        self.plan, self.action = plan, np.array([MAX_CHANGE] * len(QUEUES) + [0])

    def decide(self, snapshot: Snapshot) -> dict:
        return apply_correction(snapshot, self.plan, self.action)


class CallCenterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: CenterConfig | None = None, reward: RewardConfig | None = None,
                 scenarios: dict | None = None):
        self.config = config or CenterConfig()
        self.reward_config = reward or RewardConfig()
        self.scenarios = scenarios                    # None = random surprise days (training)
        unknown = set(self.scenarios or ()) - set(SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown scenarios: {unknown}")
        self.plan = FixedRulePolicy(self.config)
        self.action_space = gym.spaces.MultiDiscrete([2 * MAX_CHANGE + 1] * len(QUEUES) + [2])
        self.observation_space = gym.spaces.Box(0.0, 5.0, shape=(1 + 5 * len(QUEUES) + 4,), dtype=np.float32)
        self.center = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        day_seed = int(options["day"]) if "day" in options else int(self.np_random.integers(0, 2**31 - 1))
        if options.get("scenario"):
            scenario = options["scenario"]
        elif self.scenarios:
            names = list(self.scenarios)
            scenario = names[int(self.np_random.choice(len(names), p=list(self.scenarios.values())))]
        else:
            scenario = random_surprises(self.np_random)
        self.policy = _Waiting(self.plan)
        self.center = CallCenter(self.policy, seed=day_seed, config=self.config, scenario=scenario)
        self._totals = self._current_totals()
        return observation(self.center._snapshot(), self.plan, self.config), {"day": day_seed, "scenario": scenario}

    def step(self, action):
        self.policy.action = np.asarray(action)
        self.center.run_until(self.center.env.now + self.config.decision_minutes)
        totals = self._current_totals()
        staff, abandoned, late = (totals[k] - self._totals[k] for k in ("staff", "abandoned", "late"))
        self._totals = totals
        r = self.reward_config
        cost = staff + r.abandoned_call * abandoned + r.late_answer * late
        terminated = self.center.finished
        info = {"staff_cost": staff, "abandoned": abandoned, "late": late}
        if terminated:
            info["kpis"] = self.center.result.kpis()
        return observation(self.center._snapshot(), self.plan, self.config), -cost / r.scale, terminated, False, info

    def _current_totals(self) -> dict:
        res = self.center.result
        answered, within = sum(res.answered.values()), sum(res.within_sla.values())
        return {"staff": res.staff_cost, "abandoned": sum(res.abandoned.values()), "late": answered - within}


class RLPolicy(Policy):
    """A trained agent used as a normal Policy, so it runs in CallCenter like the baseline."""

    def __init__(self, model, config: CenterConfig | None = None):
        self.model, self.config = model, config or CenterConfig()
        self.plan = FixedRulePolicy(self.config)

    def decide(self, snapshot: Snapshot) -> dict:
        action, _ = self.model.predict(observation(snapshot, self.plan, self.config), deterministic=True)
        return apply_correction(snapshot, self.plan, action)

    @classmethod
    def load(cls, path, config: CenterConfig | None = None):
        from stable_baselines3 import PPO
        return cls(PPO.load(path, device="cpu"), config)


def day_cost(kpis_result, reward: RewardConfig | None = None) -> float:
    """The full cost of a day in euros, the same way the reward counts it (to compare policies)."""
    r = reward or RewardConfig()
    answered = sum(kpis_result.answered.values())
    late = answered - sum(kpis_result.within_sla.values())
    return kpis_result.staff_cost + r.abandoned_call * sum(kpis_result.abandoned.values()) + r.late_answer * late
