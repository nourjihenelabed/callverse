"""Explains the RL floor manager's decisions (XAI), with interactive charts.

Run it after training:  python -m workforce.explain
It writes HTML pages in workforce/models/explain/ (open them in a browser, no internet needed):

1. timeline_<scenario>.html  One day, every 15 minutes: advisors really on duty with the
   Erlang C plan, the reactive rule, the margin rule and the RL agent, and the calls received
   compared with the forecast (bars: above 1 means more calls than expected).
   You see exactly when the agent adds or removes advisors, and what triggered it.
2. corrections.html          On average, how many advisors the agent adds to or
   removes from the Erlang C plan, per hour and per queue, on normal and peak days.
3. what_matters.html         Which inputs drive the agent's decisions: each input is
   shuffled across many situations, and we measure how much the decisions change
   (permutation importance). A big bar = the agent relies a lot on that input.
"""

from pathlib import Path

import numpy as np

from workforce.env import RLPolicy, observation
from workforce.manager import FixedRulePolicy, MarginRulePolicy, ReactiveRulePolicy
from workforce.simulation import QUEUES, CallCenter, CenterConfig, Policy
from workforce.train import MODEL_PATH

OUT_DIR = MODEL_PATH.parent / "explain"
INPUT_NAMES = (
    ["time of day"]
    + [f"{q}: {what}" for q in QUEUES for what in ("customers waiting", "advisors working", "last avg wait", "Erlang C plan")]
    + ["last period SLA", "last period hang-ups", "forecast next hour", "priority on"]
    + [f"{q}: calls received vs forecast" for q in QUEUES]
)
COLORS = {"Erlang C plan": "#8a94a6", "Reactive rule": "#d08a2c", "Margin rule": "#3f9d6b", "RL agent": "#2f6db3"}


class Recorder(Policy):
    """Wraps a policy and writes down every situation and decision."""

    def __init__(self, policy: Policy):
        self.policy, self.log = policy, []

    def decide(self, snapshot):
        wanted = self.policy.decide(snapshot)
        self.log.append((snapshot, wanted))
        return wanted


def record_day(policy: Policy, scenario: str, day: int) -> list:
    recorder = Recorder(policy)
    CallCenter(recorder, seed=day, scenario=scenario).run()
    return recorder.log


# ---------------------------------------------------------------- 1. one day, hour by hour

def timeline(agent: RLPolicy, scenario: str = "network_outage", day: int = 10_000):
    from plotly.subplots import make_subplots

    runs = {
        "Erlang C plan": record_day(FixedRulePolicy(), scenario, day),
        "Reactive rule": record_day(ReactiveRulePolicy(), scenario, day),
        "Margin rule": record_day(MarginRulePolicy(), scenario, day),
        "RL agent": record_day(agent, scenario, day),
    }
    open_hour = CenterConfig().open_hour
    fig = make_subplots(rows=len(QUEUES), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"{q.capitalize()} queue" for q in QUEUES],
                        specs=[[{"secondary_y": True}]] * len(QUEUES))
    for row, q in enumerate(QUEUES, start=1):
        for name, log in runs.items():
            hours = [open_hour + snap.minute / 60 for snap, _ in log]
            # real staffing during each period = what the next snapshot sees (after the 1-hour minimum
            # shift, a policy cannot always send advisors home as soon as it asks)
            real = [log[k + 1][0].staffed[q] for k in range(len(log) - 1)] + [log[-1][0].staffed[q]]
            fig.add_scatter(x=hours, y=real, name=name, line_shape="hv",
                            line=dict(color=COLORS[name], width=3 if name == "RL agent" else 2),
                            legendgroup=name, showlegend=row == 1, row=row, col=1, secondary_y=False)
        log = runs["RL agent"]
        ratio = [snap.last_calls.get(q, 0) / snap.expected_calls[q] if snap.expected_calls.get(q) else None
                 for snap, _ in log]
        fig.add_bar(x=[open_hour + snap.minute / 60 for snap, _ in log], y=ratio,
                    name="Calls received ÷ forecast (last 15 min)", marker_color="rgba(208, 68, 68, 0.30)",
                    legendgroup="calls", showlegend=row == 1, row=row, col=1, secondary_y=True)
        fig.update_yaxes(title_text="advisors on duty", row=row, col=1, secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title_text="calls ÷ forecast", row=row, col=1, secondary_y=True, rangemode="tozero",
                         showgrid=False)
    fig.update_xaxes(title_text="hour of the day", row=len(QUEUES), col=1, dtick=1)
    fig.update_layout(title=f"One day, decision by decision — scenario: {scenario} (day {day})",
                      height=850, template="plotly_white", legend=dict(orientation="h", y=1.08))
    return fig


# ---------------------------------------------------------------- 2. average corrections to the plan

def corrections(agent: RLPolicy, scenarios=("normal", "network_outage", "quiet_day"), days=range(10_000, 10_010)):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plan, cfg = FixedRulePolicy(), CenterConfig()
    hours = list(range(cfg.open_hour, cfg.close_hour))
    fig = make_subplots(rows=1, cols=len(scenarios), subplot_titles=scenarios, shared_yaxes=True)
    for col, scenario in enumerate(scenarios, start=1):
        total = np.zeros((len(QUEUES), len(hours)))
        count = np.zeros(len(hours))
        for day in days:
            for snap, wanted in record_day(agent, scenario, day):
                h = int(snap.minute // 60)
                if h >= len(hours):
                    continue
                planned = plan.decide(snap)
                total[:, h] += [wanted[q] - planned[q] for q in QUEUES]
                count[h] += 1
        mean = total / np.maximum(count, 1)
        fig.add_trace(go.Heatmap(z=mean, x=hours, y=list(QUEUES), zmid=0, zmin=-3, zmax=3, colorscale="RdBu",
                                 reversescale=True, text=np.round(mean, 1), texttemplate="%{text}",
                                 showscale=col == len(scenarios), colorbar=dict(title="advisors<br>vs plan")),
                      row=1, col=col)
    fig.update_xaxes(title_text="hour", dtick=1)
    fig.update_layout(title="Advisors added (+, red) or removed (−, blue) by the RL agent compared with the Erlang C plan",
                      height=380, template="plotly_white")
    return fig


# ---------------------------------------------------------------- 3. which inputs matter

def what_matters(agents, scenarios=("normal", "network_outage", "billing_day", "afternoon_promo"),
                 days=range(10_000, 10_005), seed: int = 0):
    import plotly.graph_objects as go

    cfg, plan = CenterConfig(), FixedRulePolicy()
    situations = [snap for s in scenarios for d in days for snap, _ in record_day(agents[0], s, d)]
    X = np.stack([observation(snap, plan, cfg) for snap in situations])
    rng = np.random.default_rng(seed)
    importance = np.zeros(X.shape[1])
    for agent in agents:
        base, _ = agent.model.predict(X, deterministic=True)
        for j in range(X.shape[1]):
            shuffled = X.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            changed, _ = agent.model.predict(shuffled, deterministic=True)
            # how many advisors the decision moves on average, plus priority flips
            importance[j] += np.abs(changed - base).sum(axis=1).mean()
    importance /= len(agents)
    order = np.argsort(importance)
    fig = go.Figure(go.Bar(x=importance[order], y=[INPUT_NAMES[i] for i in order], orientation="h",
                           marker_color="#2f6db3"))
    fig.update_layout(title=f"What drives the agent's decisions (permutation importance, {len(agents)} agents, "
                            f"{len(situations)} situations)",
                      xaxis_title="average change in the decision when this input is shuffled (advisors)",
                      height=620, template="plotly_white", margin=dict(l=260))
    return fig, dict(zip(INPUT_NAMES, importance.round(3)))


def main() -> None:
    paths = sorted(MODEL_PATH.parent.glob("ppo_staffing*.zip"))
    if not paths:
        raise SystemExit("No trained agent yet. Run: python -m workforce.train")
    agents = [RLPolicy.load(p) for p in paths]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in ("normal", "network_outage", "quiet_day"):
        timeline(agents[0], scenario).write_html(OUT_DIR / f"timeline_{scenario}.html", include_plotlyjs=True)
    corrections(agents[0]).write_html(OUT_DIR / "corrections.html", include_plotlyjs=True)
    fig, scores = what_matters(agents)
    fig.write_html(OUT_DIR / "what_matters.html", include_plotlyjs=True)
    print("Inputs that matter most to the agent:")
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {score:6.3f}  {name}")
    print(f"\nCharts written to {OUT_DIR} — open the .html files in a browser.")


if __name__ == "__main__":
    main()
