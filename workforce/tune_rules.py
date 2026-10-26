"""Tunes the reactive and margin rules for the calibrated call center.

Run it after every new calibration:  python -m workforce.tune_rules
It tries every setting in a small grid on 40 random training days (the same kind
of days the RL agent trains on, never the test days of compare.py), keeps the
cheapest, and writes it to the "rules" section of workforce/calibration.json.
So the RL agent always faces the best version of each rule.
"""

import itertools
import json

import numpy as np

from workforce.env import day_cost, random_surprises
from workforce.manager import CALIBRATION_PATH, MarginRulePolicy, ReactiveRulePolicy
from workforce.simulation import CallCenter

TUNING_SEED = 123          # fixed, so everyone gets the same tuned settings
TUNING_DAYS = 40


def training_days(n: int = TUNING_DAYS, seed: int = TUNING_SEED) -> list:
    rng = np.random.default_rng(seed)
    return [(day, random_surprises(rng)) for day in range(n)]


def average_cost(policy, days) -> float:
    return sum(day_cost(CallCenter(policy, seed=day, scenario=scenario).run()) for day, scenario in days) / len(days)


def tune(days=None) -> dict:
    days = days or training_days()
    reactive = min(
        ({"per_waiting": pw, "max_extra": mx, "use_priority": prio}
         for pw, mx, prio in itertools.product((1, 2), (3, 5), (False, True))),
        key=lambda s: average_cost(ReactiveRulePolicy(**s), days),
    )
    margin = min(
        ({"margin": {"general": g, "technical": t, "complaint": c}, "priority": prio}
         for g, t, c, prio in itertools.product((1, 2, 3, 4), (0, 1, 2), (0, 1, 2), (True, False))),
        key=lambda s: average_cost(MarginRulePolicy(**s), days),
    )
    return {
        "reactive": reactive,
        "margin": margin,
        "source": f"Cheapest settings on {len(days)} random training days (seed {TUNING_SEED}), "
                  "written by python -m workforce.tune_rules.",
    }


def main() -> None:
    print(f"Tuning the reactive and margin rules on {TUNING_DAYS} training days (a few minutes)...")
    rules = tune()
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    calibration["rules"] = rules
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print("Reactive rule:", rules["reactive"])
    print("Margin rule:  ", rules["margin"])
    print(f"Wrote the 'rules' section of {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
