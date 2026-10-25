"""Compares the RL floor manager with three baselines, on normal days and on surprise peaks.

The agents train on random surprise days; the 8 named scenarios here are only used
for testing (never in training).

Run it after training:  python -m workforce.compare
Every policy gets exactly the same days (same customers, same call durations),
so the differences come only from the staffing decisions. The test days were
never used in training. If several agents were trained (--seed), the RL result
is their average, with the spread between them.
"""

import json
import math
import statistics
from pathlib import Path

from workforce.env import RLPolicy, day_cost
from workforce.manager import FixedRulePolicy, MarginRulePolicy, ReactiveRulePolicy
from workforce.simulation import ALL_SCENARIOS, CallCenter, CenterConfig
from workforce.train import MODEL_PATH

TEST_DAYS = range(10_000, 10_030)          # 30 days per scenario the agents never saw
KEYS = ("sla", "abandonment_rate", "avg_wait_s", "occupancy", "staff_cost_eur", "satisfaction")
RESULTS_PATH = Path(__file__).resolve().parent / "models" / "comparison.json"
MARKDOWN_PATH = Path(__file__).resolve().parent / "models" / "results.md"
BASELINE, REACTIVE, MARGIN, RL = "Erlang C plan", "Reactive rule", "Margin rule", "RL agent (PPO)"


def run_days(policy, scenario: str, days=TEST_DAYS, reward=None) -> tuple[dict, list]:
    """Average KPIs and the total cost of each day."""
    kpis, costs = [], []
    for day in days:
        result = CallCenter(policy, seed=day, scenario=scenario).run()
        kpis.append(result.kpis())
        costs.append(day_cost(result, reward))
    summary = {k: sum(d[k] for d in kpis) / len(kpis) for k in KEYS}
    summary["total_cost_eur"] = sum(costs) / len(costs)
    return summary, costs


def trained_agents() -> list[Path]:
    paths = sorted(MODEL_PATH.parent.glob("ppo_staffing*.zip"))
    if not paths:
        raise SystemExit("No trained agent yet. Run: python -m workforce.train")
    return paths


def paired_test(base_costs: list, other_costs: list) -> dict:
    """Is the other policy really cheaper than the baseline on the same days?"""
    diff = [b - o for b, o in zip(base_costs, other_costs)]
    mean, sd = statistics.mean(diff), statistics.stdev(diff)
    t = mean / (sd / math.sqrt(len(diff))) if sd else float("inf")
    return {"saving_per_day": round(mean, 2), "t": round(t, 2), "better_days": sum(d > 0 for d in diff),
            "days": len(diff), "significant": abs(t) >= 2.05}      # 2.05 = 95% level for 30 days


def compare(scenarios=tuple(ALL_SCENARIOS), days=TEST_DAYS, reward=None) -> dict:
    agents = [RLPolicy.load(p) for p in trained_agents()]
    results = {}
    for scenario in scenarios:
        base, base_costs = run_days(FixedRulePolicy(), scenario, days, reward)
        reactive, reactive_costs = run_days(ReactiveRulePolicy(), scenario, days, reward)
        margin, margin_costs = run_days(MarginRulePolicy(), scenario, days, reward)
        per_agent = [run_days(agent, scenario, days, reward) for agent in agents]
        rl = {k: statistics.mean(a[0][k] for a in per_agent) for k in base}
        rl_spread = {k: (statistics.stdev(a[0][k] for a in per_agent) if len(agents) > 1 else 0.0) for k in base}
        rl_costs = [statistics.mean(day) for day in zip(*(a[1] for a in per_agent))]
        results[scenario] = {
            BASELINE: _round(base),
            REACTIVE: _round(reactive),
            MARGIN: _round(margin),
            RL: _round(rl),
            "rl_spread_between_agents": _round(rl_spread),
            "agents": len(agents),
            "reactive_vs_baseline": paired_test(base_costs, reactive_costs),
            "rl_vs_baseline": paired_test(base_costs, rl_costs),
            "rl_vs_reactive": paired_test(reactive_costs, rl_costs),
            "rl_vs_margin": paired_test(margin_costs, rl_costs),
        }
    return results


def _round(d: dict) -> dict:
    return {k: round(v, 4) for k, v in d.items()}


def print_tables(results: dict) -> None:
    target = CenterConfig().sla_target
    rows = [("sla", "SLA (20 s)", "{:.1%}"), ("abandonment_rate", "Hang-ups", "{:.1%}"),
            ("avg_wait_s", "Avg wait (s)", "{:.1f}"), ("staff_cost_eur", "Staff cost (€/day)", "{:,.0f}"),
            ("total_cost_eur", "Total cost (€/day)", "{:,.0f}")]
    for scenario, r in results.items():
        print(f"\n=== {scenario} ({len(TEST_DAYS)} test days, RL = average of {r['agents']} agents)")
        print(f"{'':<20}{BASELINE:>16}{REACTIVE:>16}{MARGIN:>16}{RL:>22}")
        for key, label, fmt in rows:
            spread = r["rl_spread_between_agents"][key]
            rl_text = fmt.format(r[RL][key]) + (f" ±{fmt.format(spread)}" if r["agents"] > 1 else "")
            print(f"{label:<20}{fmt.format(r[BASELINE][key]):>16}{fmt.format(r[REACTIVE][key]):>16}"
                  f"{fmt.format(r[MARGIN][key]):>16}{rl_text:>22}")
        print(f"{'Meets SLA target':<20}" + "".join(
            f"{('yes' if r[n]['sla'] >= target else 'NO'):>{w}}"
            for n, w in ((BASELINE, 16), (REACTIVE, 16), (MARGIN, 16), (RL, 22))))
        for key, label in (("rl_vs_baseline", "RL vs Erlang C"), ("rl_vs_reactive", "RL vs reactive rule"),
                           ("rl_vs_margin", "RL vs margin rule")):
            t = r[key]
            verdict = "significant" if t["significant"] else "not significant"
            print(f"  {label}: saves {t['saving_per_day']:+,.0f} €/day, better on {t['better_days']}/{t['days']} days "
                  f"(t = {t['t']}, {verdict})")


def markdown_tables(results: dict) -> str:
    """The results as Markdown tables, ready to paste into workforce/README.md."""
    names = [BASELINE, REACTIVE, MARGIN, RL]
    label = lambda s: s.replace("_", " ")  # noqa: E731
    out = [f"Same {len(TEST_DAYS)} test days per scenario for every policy. "
           f"RL = average of {next(iter(results.values()))['agents']} agents ± the spread between them.", "",
           "### Total cost per day (staff + price of bad service)", "",
           "| Scenario | Erlang C plan | Reactive rule | Margin rule | RL agent | Best |", "|---|---|---|---|---|---|"]
    for scenario, r in results.items():
        costs = {n: r[n]["total_cost_eur"] for n in names}
        best = min(costs, key=costs.get)
        cells = []
        for n in names:
            text = f"{costs[n]:,.0f} €"
            if n == RL and r["agents"] > 1:
                text += f" ±{r['rl_spread_between_agents']['total_cost_eur']:,.0f}"
            cells.append(f"**{text}**" if n == best else text)
        out.append(f"| {label(scenario)} | " + " | ".join(cells) + f" | {best.replace(' (PPO)', '')} |")
    averages = [statistics.mean(r[n]["total_cost_eur"] for r in results.values()) for n in names]
    worst = [max(r[n]["total_cost_eur"] - min(r[m]["total_cost_eur"] for m in names) for r in results.values())
             for n in names]
    out.append("| **average** | " + " | ".join(f"{v:,.0f} €" for v in averages) + " | |")
    out.append("| **worst gap to the best policy** | " + " | ".join(f"{v:,.0f} €" for v in worst) + " | |")

    def test(t):
        verdict = "significant" if t["significant"] else "not significant"
        return f"{t['saving_per_day']:+,.0f} €/day, better on {t['better_days']}/{t['days']} ({verdict})"

    out += ["", "### Is the RL agent really better? (paired test on the same days; + = RL cheaper)", "",
            "| Scenario | RL vs Erlang C | RL vs reactive rule | RL vs margin rule |", "|---|---|---|---|"]
    out += [f"| {label(s)} | {test(r['rl_vs_baseline'])} | {test(r['rl_vs_reactive'])} | {test(r['rl_vs_margin'])} |"
            for s, r in results.items()]
    out += ["", "### Service", "",
            "| Scenario | SLA (20 s): Erlang C / reactive / margin / RL | Hang-ups: Erlang C / reactive / margin / RL |",
            "|---|---|---|"]
    out += [f"| {label(s)} | " + " / ".join(f"{r[n]['sla']:.1%}" for n in names) + " | "
            + " / ".join(f"{r[n]['abandonment_rate']:.1%}" for n in names) + " |" for s, r in results.items()]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    results = compare()
    print_tables(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    MARKDOWN_PATH.write_text(markdown_tables(results), encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH} (the dashboard can read it)")
    print(f"Tables for the README: {MARKDOWN_PATH}")
