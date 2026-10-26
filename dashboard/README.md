# Scoreboard (dashboard)

Shows the results of the pretend call center: the calls and their reviews, and the workforce manager.

## 1. Save calls

Each batch of calls is saved in two tables, `calls` and `call_scores` (`callverse/results.py`).
They are separate from the customer tables, so `scripts/seed_data.py` never erases them.

```bash
python scripts/run_calls.py --calls 50 --run night-1     # 1 to 2 hours: run it before going to bed
```

Stopped in the middle (Ctrl+C, the computer slept)? Run the same command again: the saved calls are
skipped and the next ones are the same customers, reasons and moods as they would have been.

## 2. Look at the results

**Streamlit app** (needs `pip install -e ".[dashboard]"`):

```bash
streamlit run dashboard/app.py
```

**One HTML page** (only needs Plotly; works offline, easy to send to someone):

```bash
python scripts/build_report.py               # writes dashboard/report.html
```

Both show the same charts (`dashboard/charts.py`):

| Page | What it shows |
|---|---|
| Calls | calls, share resolved, customer satisfaction, review quality, correct escalation choices; the average grade per criterion; how calls ended; quality and satisfaction by mood, queue and reason; day by day; the reviewer's most frequent tips; a table of all calls |
| One call | the conversation, the complaint specialist's decision and whether its escalation choice was right, the 6 grades and the reason for each, the tips |
| Workforce manager | the RL agent against the Erlang C plan and the two rules: average and worst cost, SLA met, cost, SLA and hang-ups per scenario (from `python -m workforce.compare`) |

## Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit app (3 pages) |
| `charts.py` | The Plotly charts, shared by the app and the report |
| `../callverse/results.py` | Saving calls, loading them, the KPIs (plain Python, no pandas) |
| `../scripts/run_calls.py` | Runs and saves a batch of calls |
| `../scripts/build_report.py` | The one-page HTML report |

## Checks

`pytest tests/test_results.py tests/test_dashboard.py`: 15 tests, including the three Streamlit pages
(skipped if Streamlit is not installed).
