"""Scoreboard: the results of the pretend call center, in the browser.

Run it from the project folder:
    streamlit run dashboard/app.py

Three pages (left menu):
- Calls: the headline numbers, the grade of each criterion, how calls ended, results by mood,
  queue and reason, day by day, and what the advisor should improve first.
- One call: the conversation, the complaint specialist's decision, and every grade with its reason.
- Workforce manager: the RL floor manager against Erlang C and the two rules (python -m workforce.compare).

The calls come from the database (saved by scripts/run_calls.py). The charts are in dashboard/charts.py,
shared with the HTML report (scripts/build_report.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from callverse import results  # noqa: E402
from callverse.state import QUALITY_CRITERIA  # noqa: E402
from dashboard import charts  # noqa: E402

st.set_page_config(page_title="CallVerse scoreboard", layout="wide")


@st.cache_data(ttl=60)
def load(run_id: str | None) -> list[dict]:
    return results.load_calls(run_id=run_id)


def percent(value) -> str:
    return "–" if value is None else f"{value:.0%}"


def grade(value) -> str:
    return "–" if value is None else f"{value:.2f}/5"


# ---------------------------------------------------------------- left menu

st.sidebar.title("CallVerse")
page = st.sidebar.radio("Page", ["Calls", "One call", "Workforce manager"])

try:
    runs = results.run_ids()
except Exception as error:          # no database: the workforce page still works
    runs, db_error = [], error
else:
    db_error = None

run = st.sidebar.selectbox("Batch of calls", ["All batches"] + runs) if runs else None
rows = load(None if run in (None, "All batches") else run) if runs else []
if st.sidebar.button("Reload"):
    st.cache_data.clear()
    st.rerun()


def no_calls() -> None:
    if db_error:
        st.error(f"The database cannot be reached: {db_error}")
    st.info("No call saved yet. Run a batch first, for example:\n\n"
            "`python scripts/run_calls.py --calls 20 --run test-1`")


# ---------------------------------------------------------------- page 1: calls

if page == "Calls":
    st.title("Calls")
    if not rows:
        no_calls()
    else:
        s = results.summary(rows)
        cols = st.columns(5)
        cols[0].metric("Calls", s["calls"])
        cols[1].metric("Resolved", percent(s["resolved_rate"]))
        cols[2].metric("Customer satisfaction", grade(s["avg_satisfaction"]))
        cols[3].metric("Quality (reviewer)", grade(s["avg_quality"]))
        cols[4].metric("Correct escalation choices", percent(s["escalation_accuracy"]),
                       help="Complaints only: the reviewer rechecks the 4 supervisor rules.")
        st.caption(f"Complaints: {percent(s['complaint_rate'])} of calls, escalated: {percent(s['escalation_rate'])}, "
                   f"average compensation {s['avg_compensation_eur'] or 0:.2f} €. Average {s['avg_turns']} customer "
                   f"messages and {s['avg_seconds']} s per call.")

        left, right = st.columns(2)
        left.plotly_chart(charts.criteria(rows), width="stretch")
        right.plotly_chart(charts.outcomes(rows), width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(charts.grouped(rows, "persona", "By customer mood"), width="stretch")
        right.plotly_chart(charts.grouped(rows, "queue", "By queue"), width="stretch")
        st.plotly_chart(charts.grouped(rows, "call_reason", "By call reason"), width="stretch")
        if len(results.daily(rows)) > 1:
            st.plotly_chart(charts.over_time(rows), width="stretch")

        st.subheader("What the advisor should improve first")
        st.caption("The reviewer's most frequent recommendations.")
        for tip, count in results.common_tips(rows):
            st.write(f"- **{count}×** {tip}")

        st.subheader("All calls")
        st.dataframe([{"call": r["call_id"], "reason": r["call_reason"], "mood": r["persona"],
                       "outcome": r["outcome"], "satisfaction": r["satisfaction"], "quality": r["overall"],
                       "turns": r["turns"], **{c: r["scores"].get(c, {}).get("score") for c in QUALITY_CRITERIA}}
                      for r in rows], width="stretch", hide_index=True)

# ---------------------------------------------------------------- page 2: one call

elif page == "One call":
    st.title("One call")
    if not rows:
        no_calls()
    else:
        labels = {f"{r['call_id']} · {r['call_reason']} · {r['persona']} · {r['outcome']} · "
                  f"{r['overall'] or 0:.1f}/5": r for r in reversed(rows)}
        row = labels[st.selectbox("Call", list(labels))]

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Conversation")
            for m in row["messages"]:
                with st.chat_message("user" if m["role"] == "client" else "assistant",
                                     avatar="🧑" if m["role"] == "client" else "🎧"):
                    st.write(m["content"])
            st.caption(f"Customer {row['customer_id']} · outcome **{row['outcome']}** · "
                       f"satisfaction {row['satisfaction']}/5 · {row['turns']} customer messages · "
                       f"tools: {', '.join(sorted({t['name'] for t in row['tools_used']})) or 'none'}")
            if row["is_complaint"]:
                st.subheader("Complaint specialist's decision")
                comp = (row["compensation_cents"] or 0) / 100
                st.write(f"Category **{row['complaint_category']}**, severity **{row['complaint_severity']}**/4, "
                         f"compensation **{comp:.2f} €**, sent to a supervisor: **{'yes' if row['escalated'] else 'no'}**, "
                         f"ticket {row['ticket_id']}.")
                if row["escalation_correct"] is not None:
                    st.write("Escalation choice checked by the reviewer: "
                             + ("✅ correct" if row["escalation_correct"] else "❌ not what the rules require"))
        with right:
            st.plotly_chart(charts.call_grades(row), width="stretch")
            st.subheader("Why these grades")
            for c in QUALITY_CRITERIA:
                g = row["scores"].get(c)
                if g:
                    with st.expander(f"{charts.CRITERIA_LABELS[c]}: {g['score']}/5", expanded=g["score"] <= 3):
                        st.write(g["explanation"])
            if row["recommendations"]:
                st.subheader("Tips for the advisor")
                for tip in row["recommendations"]:
                    st.write(f"- {tip}")

# ---------------------------------------------------------------- page 3: workforce manager

else:
    st.title("Workforce manager")
    comparison, twin = charts.load_comparison(), charts.load_twin()

    st.subheader("RL floor manager against Erlang C and the rules")
    if comparison is None:
        st.info("Not on this computer yet. Run `python scripts/run_workforce.py --skip-calibration`, "
                "or only the comparison: `python -m workforce.compare`")
    else:
        st.write("The same 30 test days per scenario for every policy (days never used in training). "
                 "Total cost = staff + the price of bad service (hang-ups, waiting).")
        st.dataframe(charts.wfm_summary(comparison), width="stretch", hide_index=True)
        st.plotly_chart(charts.wfm_metric(comparison, "total_cost_eur", "Total cost per day (€)"),
                        width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(charts.wfm_metric(comparison, "sla", "Calls answered within 20 s", percent=True),
                          width="stretch")
        right.plotly_chart(charts.wfm_metric(comparison, "abandonment_rate", "Customers who hang up", percent=True),
                           width="stretch")

    st.subheader("The loop: what the AI calls change for staffing")
    if twin is None:
        st.info("Not computed yet. After a batch of calls, run: `python -m workforce.twin --run night-1`")
    else:
        m = twin["measured"]
        st.write(f"Measured on **{m['calls']} AI calls** (batch `{twin.get('run', 'all')}`): "
                 f"**{m['fcr']:.0%}** resolved at the first contact. If {twin['callback_share']:.0%} of "
                 f"unresolved customers call back, the center gets **{twin['load']['volume_factor'] - 1:+.0%} "
                 f"calls**, and calls that became complaints move to the complaint queue. Call durations "
                 f"stay those of the real Technion data.")
        st.dataframe(charts.twin_measured_rows(twin), width="stretch", hide_index=True)
        st.dataframe(charts.twin_rows(twin), width="stretch", hide_index=True)
        st.plotly_chart(charts.twin_costs(twin), width="stretch")
        names = list(twin["results"])
        saving = twin["results"][names[2]]["total_cost_eur"] - twin["results"][names[3]]["total_cost_eur"]
        st.success(f"A better advisor ({twin['target_fcr']:.0%} resolved first time) would save "
                   f"**{saving:,} € per day** in this center.")

    explain = charts.COMPARISON_PATH.parent / "explain"
    if explain.exists():
        st.subheader("Why the RL agent decides what it decides")
        st.write("Open these pages in your browser (made by `python -m workforce.explain`):")
        for page_file in sorted(explain.glob("*.html")):
            st.code(str(page_file), language=None)
