"""Scoreboard: the charts, as Plotly figures, shared by the Streamlit app (dashboard/app.py)
and the single-page HTML report (scripts/build_report.py).

Plain Python and Plotly only (no pandas). Each function takes the rows of
callverse.results.load_calls() or the workforce comparison, and returns a figure.
"""

import json
from pathlib import Path

import plotly.graph_objects as go

from callverse.results import by, daily, summary
from callverse.state import QUALITY_CRITERIA

TEMPLATE = "plotly_white"
BLUE, GREEN, ORANGE, RED, GREY = "#2f6db3", "#3f9d6b", "#d08a2c", "#c0392b", "#8a94a6"
OUTCOME_COLORS = {"resolved": GREEN, "escalated": BLUE, "refused": ORANGE, "abandoned": RED}
CRITERIA_LABELS = {"relevance": "Relevance", "accuracy": "Accuracy", "compliance": "Compliance",
                   "empathy": "Empathy", "resolution": "Resolution", "satisfaction": "Satisfaction"}

COMPARISON_PATH = Path(__file__).resolve().parent.parent / "workforce" / "models" / "comparison.json"
WFM_POLICIES = ("Erlang C plan", "Reactive rule", "Margin rule", "RL agent (PPO)")
WFM_COLORS = {"Erlang C plan": GREY, "Reactive rule": ORANGE, "Margin rule": GREEN, "RL agent (PPO)": BLUE}


def _layout(fig: go.Figure, title: str, height: int = 360, **extra) -> go.Figure:
    fig.update_layout(title=title, template=TEMPLATE, height=height, margin=dict(l=40, r=20, t=60, b=40),
                      legend=dict(orientation="h", y=-0.2), **extra)
    return fig


# ---------------------------------------------------------------- calls

def criteria(rows: list[dict]) -> go.Figure:
    """Average grade of each of the 6 criteria, from 1 to 5."""
    s = summary(rows)["criteria"]
    names = [CRITERIA_LABELS[c] for c in QUALITY_CRITERIA]
    values = [s[c] or 0 for c in QUALITY_CRITERIA]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=BLUE, text=[f"{v:.1f}" for v in values],
                           textposition="outside"))
    return _layout(fig, "Average grade per criterion (1 to 5)", yaxis=dict(range=[0, 5.5]))


def outcomes(rows: list[dict]) -> go.Figure:
    counts = summary(rows)["outcomes"]
    order = [o for o in OUTCOME_COLORS if o in counts] + [o for o in counts if o not in OUTCOME_COLORS]
    fig = go.Figure(go.Bar(x=order, y=[counts[o] for o in order], text=[counts[o] for o in order],
                           textposition="outside", marker_color=[OUTCOME_COLORS.get(o, GREY) for o in order]))
    return _layout(fig, "How the calls ended")


def grouped(rows: list[dict], key: str, title: str) -> go.Figure:
    """Average quality and customer satisfaction for each persona, queue or reason."""
    groups = by(rows, key)
    names = list(groups)
    fig = go.Figure([
        go.Bar(name="Quality (reviewer)", x=names, y=[groups[n]["avg_quality"] or 0 for n in names], marker_color=BLUE),
        go.Bar(name="Satisfaction (customer)", x=names, y=[groups[n]["avg_satisfaction"] or 0 for n in names],
               marker_color=GREEN),
    ])
    fig.update_traces(customdata=[[groups[n]["calls"]] for n in names],
                      hovertemplate="%{x}: %{y:.2f}/5 (%{customdata[0]} calls)<extra>%{fullData.name}</extra>")
    return _layout(fig, title, barmode="group", yaxis=dict(range=[0, 5.5]))


def over_time(rows: list[dict]) -> go.Figure:
    days = daily(rows)
    x = list(days)
    fig = go.Figure([
        go.Scatter(name="Quality (reviewer)", x=x, y=[days[d]["avg_quality"] for d in x], mode="lines+markers",
                   line_color=BLUE),
        go.Scatter(name="Satisfaction (customer)", x=x, y=[days[d]["avg_satisfaction"] for d in x],
                   mode="lines+markers", line_color=GREEN),
        go.Bar(name="Resolved share × 5", x=x, y=[(days[d]["resolved_rate"] or 0) * 5 for d in x],
               marker_color="rgba(63,157,107,0.18)"),
    ])
    return _layout(fig, "Day by day", yaxis=dict(range=[0, 5.5]))


def call_grades(row: dict) -> go.Figure:
    """The 6 grades of one call."""
    names = [CRITERIA_LABELS[c] for c in QUALITY_CRITERIA]
    values = [row["scores"].get(c, {}).get("score", 0) for c in QUALITY_CRITERIA]
    colors = [GREEN if v >= 4 else ORANGE if v == 3 else RED for v in values]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors, text=values,
                           textposition="outside"))
    return _layout(fig, f"Grades of this call (overall {row['overall'] or 0:.1f}/5)", height=300,
                   xaxis=dict(range=[0, 5.5]), yaxis=dict(autorange="reversed"))


# ---------------------------------------------------------------- workforce manager

def load_comparison(path: Path = COMPARISON_PATH) -> dict | None:
    """The results of python -m workforce.compare, or None if it has not been run on this computer."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def wfm_metric(comparison: dict, key: str, title: str, percent: bool = False) -> go.Figure:
    scenarios = list(comparison)
    labels = [s.replace("_", " ") for s in scenarios]
    fig = go.Figure()
    for policy in WFM_POLICIES:
        values = [comparison[s][policy][key] for s in scenarios]
        error = None
        if policy.startswith("RL") and comparison[scenarios[0]].get("agents", 1) > 1:
            error = dict(type="data", array=[comparison[s]["rl_spread_between_agents"][key] for s in scenarios])
        fig.add_trace(go.Bar(name=policy, x=labels, y=values, marker_color=WFM_COLORS[policy], error_y=error))
    if percent:
        fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title, height=400, barmode="group")


def wfm_summary(comparison: dict) -> list[dict]:
    """One line per policy: average and worst total cost over the scenarios, SLA met or not."""
    from workforce.simulation import CenterConfig

    target = CenterConfig().sla_target
    lines = []
    for policy in WFM_POLICIES:
        costs = [comparison[s][policy]["total_cost_eur"] for s in comparison]
        best = [min(comparison[s][p]["total_cost_eur"] for p in WFM_POLICIES) for s in comparison]
        lines.append({
            "Policy": policy,
            "Average cost (€/day)": round(sum(costs) / len(costs)),
            "Worst gap to the best (€/day)": round(max(c - b for c, b in zip(costs, best))),
            "Scenarios meeting the SLA": f"{sum(comparison[s][policy]['sla'] >= target for s in comparison)}"
                                         f"/{len(comparison)}",
        })
    return lines


# ---------------------------------------------------------------- the digital twin loop (phase 3)

TWIN_PATH = COMPARISON_PATH.parent / "twin.json"


def load_twin(path: Path = TWIN_PATH) -> dict | None:
    """The results of python -m workforce.twin, or None if it has not been run on this computer."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def twin_costs(report: dict) -> go.Figure:
    names = list(report["results"])
    values = [report["results"][n]["total_cost_eur"] for n in names]
    colors = [GREY, RED, BLUE, GREEN][:len(names)]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:,} €" for v in values], textposition="outside"))
    return _layout(fig, "Total cost per day on the same simulated days (€)", height=320,
                   yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(values) * 1.2]))


def twin_rows(report: dict) -> list[dict]:
    return [{"Situation": name, "Calls / day": r["calls"], "SLA (20 s)": f"{r['sla']:.1%}",
             "Hang-ups": f"{r['abandonment_rate']:.1%}", "Staff cost (€)": r["staff_cost_eur"],
             "Total cost (€)": r["total_cost_eur"]} for name, r in report["results"].items()]


def twin_measured_rows(report: dict) -> list[dict]:
    load = report["load"]
    return [{"Queue": q, "AI calls": v["calls"], "Resolved first time": "–" if v["fcr"] is None else f"{v['fcr']:.0%}",
             "Became complaints": f"{v['to_complaint']:.0%}", "Calls back (factor)": load["repeat_factor"][q],
             "Share of calls after": f"{load['call_mix'][q]:.0%}"}
            for q, v in report["measured"]["queues"].items()]
