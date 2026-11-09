"""Phase 3: the conversations change the staffing. This closes the loop of the digital twin.

The AI calls (saved by scripts/run_calls.py) measure two things the Technion data cannot:
1. First-contact resolution (FCR), per queue: the share of calls that end resolved or handed to a
   supervisor. A customer who hangs up or refuses the answer often calls back: more calls tomorrow.
2. Calls that become complaints: a general or technical call where the advisor detects a complaint
   goes to the complaint specialist, so it is work for the complaint-skilled advisors.

The call durations and the arrival pattern stay those of the real Technion data (calibration.json):
the twin changes how many calls there are and where they go, not how long they last.

    measured = measure(load_calls("night-1"))
    config = twin_config(measured)              # a CenterConfig for the workforce simulation

Run the whole comparison with: python -m workforce.twin --run night-1
Then train an RL agent on the twin's floor (optional): python -m workforce.train --twin
"""

import argparse
import json
import statistics
from pathlib import Path

from workforce.simulation import QUEUES, CenterConfig, Policy

RESOLVED = ("resolved", "escalated")      # escalated: a supervisor calls back, the customer does not
CALLBACK_SHARE = 0.5                       # an unresolved customer calls back once, half the time (assumption)
MIN_CALLS = 5                              # fewer calls than this in a queue: keep the neutral value
TWIN_PATH = Path(__file__).resolve().parent / "models" / "twin.json"
# the RL agent retrained on the twin's floor (python -m workforce.train --twin). Not named ppo_staffing*:
# compare.py and explain.py average the agents trained on the bank floor, and must not pick this one.
TWIN_MODEL_PATH = Path(__file__).resolve().parent / "models" / "ppo_twin.zip"


# ---------------------------------------------------------------- 1. measure the calls

def measure(rows: list[dict]) -> dict:
    """What the AI calls say about each queue (rows from callverse.results.load_calls)."""
    per_queue = {}
    for queue in QUEUES:
        calls = [r for r in rows if r["queue"] == queue]
        n = len(calls)
        resolved = sum(r["outcome"] in RESOLVED for r in calls)
        to_complaint = sum(bool(r["is_complaint"]) for r in calls) if queue != "complaint" else 0
        per_queue[queue] = {
            "calls": n,
            "fcr": round(resolved / n, 3) if n else None,
            "to_complaint": round(to_complaint / n, 3) if n and queue != "complaint" else 0.0,
            "avg_turns": round(statistics.mean(r["turns"] for r in calls), 2) if n else None,
        }
    total = len(rows)
    return {
        "calls": total,
        "fcr": round(sum(r["outcome"] in RESOLVED for r in rows) / total, 3) if total else None,
        "queues": per_queue,
    }


def usable(measured: dict, queue: str, key: str, neutral: float) -> float:
    q = measured["queues"][queue]
    return q[key] if q["calls"] >= MIN_CALLS and q[key] is not None else neutral


# ---------------------------------------------------------------- 2. turn it into call-center settings

def twin_load(measured: dict, base: CenterConfig | None = None, callback_share: float = CALLBACK_SHARE,
              fcr: dict | None = None) -> dict:
    """How the calls of each queue change. `fcr` replaces the measured FCR (for "what if" scenarios).

    Returns {"volume_factor", "call_mix", "repeat_factor": {queue: ...}}."""
    base = base or CenterConfig()
    mix, new = base.call_mix, {}
    repeat = {}
    for queue in QUEUES:
        rate = (fcr or {}).get(queue, usable(measured, queue, "fcr", 1.0))
        repeat[queue] = 1 + (1 - rate) * callback_share
    moved = 0.0
    for queue in ("general", "technical"):
        share = usable(measured, queue, "to_complaint", 0.0)
        calls = mix[queue] * repeat[queue]
        new[queue] = calls * (1 - share)
        moved += calls * share
    new["complaint"] = mix["complaint"] * repeat["complaint"] + moved
    volume = sum(new.values())
    return {"volume_factor": round(volume, 4), "call_mix": {q: round(new[q] / volume, 4) for q in QUEUES},
            "repeat_factor": {q: round(r, 4) for q, r in repeat.items()}}


def what_if_fcr(measured: dict, target: float) -> dict:
    """Per queue: the measured FCR, raised to `target` where it is lower (a better advisor)."""
    return {q: max(target, usable(measured, q, "fcr", 1.0)) for q in QUEUES}


def saved_measured() -> dict | None:
    """The measurement saved by the last `python -m workforce.twin` (no database needed)."""
    try:
        return json.loads(TWIN_PATH.read_text(encoding="utf-8"))["measured"]
    except (FileNotFoundError, KeyError, ValueError):
        return None


class OldForecast(Policy):
    """Runs a floor manager on the twin's floor while it still believes the old (bank) forecast.

    Inside the simulation the forecast comes from the twin's settings, so a policy reading it would
    "know" about the repeat calls. This wrapper shows it the forecast it was planned with instead:
    the calls it then sees above that forecast are the surprise it has to notice by itself."""

    def __init__(self, policy, believed: CenterConfig, actual: CenterConfig):
        self.policy, self.believed, self.actual = policy, believed, actual

    def decide(self, snapshot):
        from dataclasses import replace

        hour = min(int(snapshot.minute // 60), len(self.actual.calls_per_hour) - 1)
        actual_rate, believed_rate = self.actual.calls_per_hour[hour], self.believed.calls_per_hour[hour]
        ratio = believed_rate / actual_rate if actual_rate else 1.0
        expected = {q: snapshot.expected_calls.get(q, 0) * ratio * self.believed.call_mix[q]
                    / (self.actual.call_mix[q] or 1) for q in QUEUES}
        return self.policy.decide(replace(snapshot, calls_next_hour=snapshot.calls_next_hour * ratio,
                                          expected_calls=expected))


def load_agent(path: Path, config: CenterConfig):
    """A trained agent as a policy, or None (RL packages not installed, or no model file)."""
    try:
        from workforce.env import RLPolicy
        return RLPolicy.load(path, config) if path.exists() else None
    except Exception:
        return None


def twin_config(measured: dict, base: CenterConfig | None = None, **load_options) -> CenterConfig:
    """The calibrated center, with the number and mix of calls the AI conversations imply."""
    base = base or CenterConfig()
    load = twin_load(measured, base, **load_options)
    return CenterConfig(
        calls_per_hour=tuple(c * load["volume_factor"] for c in base.calls_per_hour),
        call_mix=load["call_mix"],
        mean_handle_minutes=dict(base.mean_handle_minutes),
        handle_spread=base.handle_spread,
        handle_spread_by_queue=dict(base.handle_spread_by_queue),
        mean_patience_minutes=dict(base.mean_patience_minutes),
        mean_patience_by_hour=base.mean_patience_by_hour,
        forecast_error=base.forecast_error,
        roster=base.roster,
    )


# ---------------------------------------------------------------- 3. what it means for staffing

def day_cost(result) -> float:
    """The cost of a day as the RL reward counts it: staff + 50 € per hang-up + 8 € per late answer.
    Uses workforce.env when the RL packages are installed; the same formula otherwise, so the twin
    also runs (and is tested) without them."""
    try:
        from workforce.env import day_cost as reward_cost
    except ImportError:
        answered = sum(result.answered.values())
        late = answered - sum(result.within_sla.values())
        return result.staff_cost + 50.0 * sum(result.abandoned.values()) + 8.0 * late
    return reward_cost(result)


def run_days(policy, config: CenterConfig, days) -> dict:
    from workforce.simulation import CallCenter

    kpis, costs = [], []
    for day in days:
        result = CallCenter(policy, seed=day, config=config).run()
        kpis.append(result.kpis())
        costs.append(day_cost(result))
    mean = lambda key: statistics.mean(k[key] for k in kpis)  # noqa: E731
    return {"calls": round(mean("calls")), "sla": round(mean("sla"), 4),
            "abandonment_rate": round(mean("abandonment_rate"), 4),
            "staff_cost_eur": round(mean("staff_cost_eur")), "total_cost_eur": round(statistics.mean(costs))}


def compare(measured: dict, days=range(20_000, 20_020), target_fcr: float = 0.8,
            callback_share: float = CALLBACK_SHARE, with_rl: bool = True) -> dict:
    """The same days, for: the center as calibrated, the center the AI calls imply (planned with the old
    forecast, then with the new one), and the center if the advisor resolved `target_fcr` of calls.
    With the RL packages and trained agents present, two more rows: the agent trained on the bank floor,
    which does not know about the repeat calls, and the agent retrained on the twin's floor."""
    from workforce.manager import FixedRulePolicy
    from workforce.train import MODEL_PATH

    base = CenterConfig()
    twin = twin_config(measured, base, callback_share=callback_share)
    better_fcr = what_if_fcr(measured, target_fcr)
    better = twin_config(measured, base, callback_share=callback_share, fcr=better_fcr)
    results = {
        "As calibrated (Technion only)": run_days(FixedRulePolicy(base), base, days),
        "AI calls, plan ignores them": run_days(FixedRulePolicy(base), twin, days),
        "AI calls, plan uses them": run_days(FixedRulePolicy(twin), twin, days),
        f"Advisor at {target_fcr:.0%} FCR, plan uses it": run_days(FixedRulePolicy(better), better, days),
    }
    if with_rl:
        bank_agent = load_agent(MODEL_PATH, base)
        if bank_agent is not None:
            results["RL agent trained on bank data (does not know)"] = run_days(
                OldForecast(bank_agent, base, twin), twin, days)
        twin_agent = load_agent(TWIN_MODEL_PATH, twin)
        if twin_agent is not None:
            results["RL agent retrained on the twin"] = run_days(twin_agent, twin, days)
    return {
        "measured": measured,
        "load": twin_load(measured, base, callback_share=callback_share),
        "load_better": twin_load(measured, base, callback_share=callback_share, fcr=better_fcr),
        "target_fcr": target_fcr,
        "callback_share": callback_share,
        "base_calls_per_day": round(sum(base.calls_per_hour)),
        "results": results,
    }


def print_report(report: dict) -> None:
    m, load = report["measured"], report["load"]
    print(f"\nMeasured on {m['calls']} AI calls: first-contact resolution {m['fcr']:.0%}")
    for q, v in m["queues"].items():
        fcr = "–" if v["fcr"] is None else f"{v['fcr']:.0%}"
        print(f"  {q:<10} {v['calls']:>3} calls   FCR {fcr:>4}   became complaints {v['to_complaint']:.0%}")
    print(f"\nWith {report['callback_share']:.0%} of unresolved customers calling back: "
          f"{load['volume_factor'] - 1:+.0%} calls, mix "
          + ", ".join(f"{q} {s:.0%}" for q, s in load["call_mix"].items()))
    print(f"\n{'':<38}{'calls/day':>10}{'SLA':>8}{'hang-ups':>10}{'staff €':>9}{'total €':>9}")
    for name, r in report["results"].items():
        print(f"{name:<38}{r['calls']:>10}{r['sla']:>8.1%}{r['abandonment_rate']:>10.1%}"
              f"{r['staff_cost_eur']:>9,}{r['total_cost_eur']:>9,}")
    names = list(report["results"])
    saving = report["results"][names[2]]["total_cost_eur"] - report["results"][names[3]]["total_cost_eur"]
    print(f"\nA better advisor ({report['target_fcr']:.0%} first-contact resolution) saves {saving:,} € per day.")


def main() -> None:
    from callverse.results import load_calls

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="the batch of AI calls to use; all saved calls if not given")
    parser.add_argument("--target-fcr", type=float, default=0.8)
    parser.add_argument("--callback-share", type=float, default=CALLBACK_SHARE,
                        help="share of unresolved customers who call back (an assumption)")
    args = parser.parse_args()

    rows = load_calls(run_id=args.run)
    if not rows:
        raise SystemExit("No saved calls. Run: python scripts/run_calls.py --calls 50 --run night-1")
    report = compare(measure(rows), target_fcr=args.target_fcr, callback_share=args.callback_share)
    report["run"] = args.run or "all"
    print_report(report)
    TWIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TWIN_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved to {TWIN_PATH} (the dashboard shows it)")


if __name__ == "__main__":
    main()
