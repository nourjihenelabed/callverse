"""The scoreboard as one HTML page: open it in any browser, send it to someone, no server needed.

Run it from the project folder:
    python scripts/build_report.py                  # all saved calls
    python scripts/build_report.py --run night-1    # one batch

It writes dashboard/report.html with the same charts as the Streamlit app (dashboard/charts.py):
the headline numbers, the grades per criterion, results by mood, queue and reason, the reviewer's
most frequent tips, every call with its conversation and the reasons for its grades, and the
workforce manager's comparison if python -m workforce.compare has been run.
"""

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.io as pio  # noqa: E402

from callverse import results  # noqa: E402
from callverse.state import QUALITY_CRITERIA  # noqa: E402
from dashboard import charts  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "dashboard" / "report.html"

STYLE = """
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0 auto; max-width: 1150px;
       padding: 24px; color: #1f2933; background: #fff; }
h1 { margin-bottom: 4px; } h2 { margin-top: 40px; border-bottom: 1px solid #e4e7eb; padding-bottom: 6px; }
.muted { color: #6b7280; } .tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.tile { border: 1px solid #e4e7eb; border-radius: 8px; padding: 12px 16px; min-width: 150px; }
.tile b { display: block; font-size: 26px; margin-top: 4px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
@media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border-bottom: 1px solid #e4e7eb; padding: 6px 8px; text-align: left; }
details { border: 1px solid #e4e7eb; border-radius: 8px; margin: 8px 0; padding: 8px 12px; }
summary { cursor: pointer; font-weight: 600; }
.msg { margin: 6px 0; padding: 8px 10px; border-radius: 8px; }
.client { background: #f3f4f6; } .advisor { background: #eaf1fb; margin-left: 40px; }
.grade { margin: 4px 0; } .low { color: #c0392b; font-weight: 600; }
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def figure(fig, first: bool) -> str:
    """The chart as HTML. Plotly's script is put once, in the first chart, so the page works offline."""
    return pio.to_html(fig, full_html=False, include_plotlyjs="inline" if first else False,
                       config={"displayModeBar": False})


def tiles(s: dict) -> str:
    def pct(v):
        return "–" if v is None else f"{v:.0%}"

    def grade(v):
        return "–" if v is None else f"{v:.2f}/5"

    items = [("Calls", s["calls"]), ("Resolved", pct(s["resolved_rate"])),
             ("Customer satisfaction", grade(s["avg_satisfaction"])), ("Quality (reviewer)", grade(s["avg_quality"])),
             ("Correct escalation choices", pct(s["escalation_accuracy"]))]
    return '<div class="tiles">' + "".join(f'<div class="tile">{esc(k)}<b>{esc(v)}</b></div>' for k, v in items) + "</div>"


def call_block(row: dict) -> str:
    messages = "".join(f'<div class="msg {esc(m["role"])}"><b>{"Client" if m["role"] == "client" else "Conseiller"} :</b> '
                       f'{esc(m["content"])}</div>' for m in row["messages"])
    grades = "".join(
        f'<div class="grade"><span class="{"low" if g["score"] <= 3 else ""}">'
        f'{esc(charts.CRITERIA_LABELS[c])} {g["score"]}/5</span> — {esc(g["explanation"])}</div>'
        for c in QUALITY_CRITERIA if (g := row["scores"].get(c)))
    complaint = ""
    if row["is_complaint"]:
        correct = {True: " · escalation choice ✅ correct", False: " · escalation choice ❌ not what the rules require",
                   None: ""}[row["escalation_correct"]]
        complaint = (f'<p><b>Complaint specialist:</b> {esc(row["complaint_category"])}, severity '
                     f'{esc(row["complaint_severity"])}/4, compensation {(row["compensation_cents"] or 0) / 100:.2f} €, '
                     f'supervisor: {"yes" if row["escalated"] else "no"}{correct}</p>')
    tips = "".join(f"<li>{esc(t)}</li>" for t in row["recommendations"])
    return (f'<details><summary>{esc(row["call_id"])} · {esc(row["call_reason"])} · {esc(row["persona"])} · '
            f'{esc(row["outcome"])} · satisfaction {esc(row["satisfaction"])}/5 · quality {row["overall"] or 0:.1f}/5'
            f'</summary>{messages}{complaint}<h4>Why these grades</h4>{grades}'
            + (f"<h4>Tips</h4><ul>{tips}</ul>" if tips else "") + "</details>")


def build(rows: list[dict], comparison: dict | None, title: str, include_twin: bool = True) -> str:
    parts, first = [], True

    def add(fig):
        nonlocal first
        parts.append(figure(fig, first))
        first = False

    parts.append(f"<h1>{esc(title)}</h1><p class='muted'>Made on {datetime.now():%d/%m/%Y %H:%M}.</p>")
    if rows:
        s = results.summary(rows)
        parts.append("<h2>Calls</h2>" + tiles(s))
        parts.append('<div class="grid">')
        add(charts.criteria(rows)); add(charts.outcomes(rows))                            # noqa: E702
        add(charts.grouped(rows, "persona", "By customer mood")); add(charts.grouped(rows, "queue", "By queue"))  # noqa: E702
        parts.append("</div>")
        add(charts.grouped(rows, "call_reason", "By call reason"))
        if len(results.daily(rows)) > 1:
            add(charts.over_time(rows))
        tips = "".join(f"<li><b>{n}×</b> {esc(t)}</li>" for t, n in results.common_tips(rows))
        parts.append(f"<h2>What the advisor should improve first</h2><ul>{tips}</ul>")
        parts.append("<h2>Every call</h2>" + "".join(call_block(r) for r in reversed(rows)))
    else:
        parts.append("<p>No call saved yet: run <code>python scripts/run_calls.py --calls 20 --run test-1</code>.</p>")

    if comparison:
        table = charts.wfm_summary(comparison)
        head = "".join(f"<th>{esc(k)}</th>" for k in table[0])
        body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in line.values()) + "</tr>" for line in table)
        parts.append("<h2>Workforce manager</h2><p>The same 30 test days per scenario for every policy. "
                     f"Total cost = staff + the price of bad service.</p><table><tr>{head}</tr>{body}</table>")
        add(charts.wfm_metric(comparison, "total_cost_eur", "Total cost per day (€)"))
        parts.append('<div class="grid">')
        add(charts.wfm_metric(comparison, "sla", "Calls answered within 20 s", percent=True))
        add(charts.wfm_metric(comparison, "abandonment_rate", "Customers who hang up", percent=True))
        parts.append("</div>")

    twin = charts.load_twin() if include_twin else None
    if twin:
        rows_twin = charts.twin_rows(twin)
        head = "".join(f"<th>{esc(k)}</th>" for k in rows_twin[0])
        body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in line.values()) + "</tr>" for line in rows_twin)
        m = twin["measured"]
        parts.append(f"<h2>The loop: what the AI calls change for staffing</h2><p>Measured on {m['calls']} AI calls: "
                     f"{m['fcr']:.0%} resolved at the first contact, so {twin['load']['volume_factor'] - 1:+.0%} calls "
                     f"if {twin['callback_share']:.0%} of unresolved customers call back.</p>"
                     f"<table><tr>{head}</tr>{body}</table>")
        add(charts.twin_costs(twin))

    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style>'
            f'</head><body>{"".join(parts)}</body></html>')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="one batch only; all saved calls if not given")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = results.load_calls(run_id=args.run)
    title = "CallVerse scoreboard" + (f" — {args.run}" if args.run else "")
    args.out.write_text(build(rows, charts.load_comparison(), title), encoding="utf-8")
    print(f"{len(rows)} calls. Open this file in your browser: {args.out}")


if __name__ == "__main__":
    main()
