"""CallVerse Studio: the virtual call center in the browser.

Run it from the project folder, with Ollama running and the database filled:

    python studio/server.py              # then open http://127.0.0.1:8600
    python studio/server.py --port 8700  # another port

What it does:
- Live floor: place a call (customer, reason, mood) and watch it happen. Every agent lights up while it
  works, the conversation appears line by line, and the case file (CallState) fills in: what the advisor
  understood, the complaint specialist's decision, the supervisor's offer, the outcome and the review.
  The call goes through the real LangGraph graph (callverse/graph.py), streamed with stream_mode "tasks"
  (an agent starts) and "updates" (what it changed).
- Call history: replay any saved call (from run_calls.py, or from the studio itself: batch "studio").
- Results: the KPIs of a batch, the workforce comparison and the digital-twin report.

Only the Python standard library is used for the web server, so there is nothing new to install.
"""

import argparse
import json
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from callverse.state import CallState  # noqa: E402

PAGE = Path(__file__).resolve().parent / "index.html"
MODELS = ROOT / "workforce" / "models"
STUDIO_RUN = "studio"
DEMO_CUSTOMER = "9237-HQITU"       # high churn risk: shows the retention supervisor


# ---------------------------------------------------------------- turning objects into JSON

def jsonable(value):
    """Pydantic models, dates and containers as plain JSON values."""
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


# ---------------------------------------------------------------- one live call

def call_events(state: CallState, save: bool = True):
    """Runs one call through the real graph and yields what happens, as small dicts:
    start (an agent begins), update (what it changed), done (the finished case file), error."""
    from callverse.graph import _max_turns, build_graph

    started = time.time()
    last = None
    try:
        stream = build_graph().stream(state, {"recursion_limit": 4 * _max_turns() + 10},
                                      stream_mode=["tasks", "updates", "values"])
        for mode, data in stream:
            seconds = round(time.time() - started, 1)
            if mode == "tasks" and "input" in data:               # a task starts (its result comes later)
                yield {"type": "start", "node": data["name"], "t": seconds}
            elif mode == "updates":
                for node, update in data.items():
                    yield {"type": "update", "node": node, "t": seconds, "data": jsonable(update or {})}
            elif mode == "values":
                last = data
    except Exception as error:                                   # Ollama down, model missing, database...
        yield {"type": "error", "message": f"{type(error).__name__}: {error}",
               "detail": traceback.format_exc(limit=3)}
        return

    done = CallState.model_validate(last)
    seconds = round(time.time() - started, 1)
    saved = False
    if save:
        try:
            from callverse.results import save_call
            save_call(done, STUDIO_RUN, seconds)
            saved = True
        except Exception:                                        # the call still counts on screen
            saved = False
    yield {"type": "done", "t": seconds, "saved": saved, "state": jsonable(done)}


# ---------------------------------------------------------------- data for the page

def options() -> dict:
    from callverse.agents.client.personas import PERSONAS
    from callverse.agents.client.scenarios import REASONS
    return {
        "reasons": [{"id": k, "queue": r.queue, "needs": r.needs} for k, r in REASONS.items()],
        "personas": [{"id": k, "description": m.description, "patience": m.patience} for k, m in PERSONAS.items()],
        "customers": customer_list(),
        "demo_customer": DEMO_CUSTOMER,
    }


def customer_list(limit: int = 14) -> list[dict]:
    """A few customers to pick from: the demo customer, the highest churn risks, and some ordinary ones."""
    from sqlalchemy import select

    from callverse.db import customers, get_engine
    with get_engine().connect() as conn:
        risky = list(conn.execute(select(customers.c.customer_id).order_by(customers.c.churn_risk.desc())
                                  .limit(5)).scalars())
        calm = list(conn.execute(select(customers.c.customer_id).where(customers.c.churn_risk < 0.3)
                                 .order_by(customers.c.customer_id).limit(limit - 6)).scalars())
    ids = list(dict.fromkeys([DEMO_CUSTOMER, *risky, *calm]))
    return [c for c in (customer_card(i) for i in ids) if c]


def customer_card(customer_id: str) -> dict | None:
    from callverse.agents.client.scenarios import customer_facts
    try:
        f = customer_facts(customer_id)
    except ValueError:
        return None
    return {"id": customer_id, "offers": f["offers"], "codes": f["codes"], "bill": f["bill"],
            "contract": f["contract"], "payment": f["payment"], "tenure": f["tenure"],
            "churn_risk": round(f["churn_risk"] or 0, 2)}


def health() -> dict:
    """Is everything needed for a live call running?"""
    import os

    from callverse.config import load_settings
    from callverse.llm import DEFAULT_OLLAMA_URL
    settings = load_settings()
    status = {"model": settings["llm"]["model"], "database": False, "ollama": False, "model_ready": False}
    try:
        from sqlalchemy import text

        from callverse.db import get_engine
        with get_engine().connect() as conn:
            conn.execute(text("select 1"))
        status["database"] = True
    except Exception as error:
        status["database_error"] = str(error).splitlines()[0][:200]
    url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    status["ollama_url"] = url
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=3) as reply:
            names = [m["name"] for m in json.loads(reply.read())["models"]]
        status["ollama"] = True
        status["model_ready"] = settings["llm"]["model"] in names
    except Exception as error:
        status["ollama_error"] = str(error)[:200]
    return status


def calls_list(run_id: str | None) -> list[dict]:
    from callverse.results import load_calls
    keep = ("call_id", "run_id", "customer_id", "persona", "call_reason", "queue", "outcome", "satisfaction",
            "overall", "turns", "seconds", "is_complaint", "wants_to_leave", "finished_at")
    rows = load_calls(run_id=run_id)
    return [jsonable({k: r.get(k) for k in keep}) for r in reversed(rows)]


def one_call(call_id: str) -> dict | None:
    from callverse.results import load_calls
    return next((jsonable(r) for r in load_calls() if r["call_id"] == call_id), None)


def stats(run_id: str | None) -> dict:
    from callverse.results import by, common_tips, load_calls, run_ids, summary
    rows = load_calls(run_id=run_id)
    return {"runs": run_ids(), "run": run_id, "summary": summary(rows), "by_persona": by(rows, "persona"),
            "by_queue": by(rows, "queue"), "tips": common_tips(rows)}


SCENARIO_TEXT = {
    "normal": "The usual day, as forecast.",
    "network_outage": "Technical calls ×2 from 10h to 13h, complaints ×1.5 from 11h to 14h.",
    "billing_day": "Bills are sent: general ×1.3 and complaints ×1.6 all day.",
    "afternoon_promo": "An advertising campaign: general calls ×1.8 from 14h to 17h.",
    "late_outage": "Never seen in training: the outage, but from 16h.",
    "big_outage": "Never seen in training: technical ×2.5 from 10h to 13h, complaints ×1.8 from 11h to 14h.",
    "complaint_storm": "Never seen in training: complaints ×2.5 from 9h to 12h.",
    "quiet_day": "Never seen in training: 30% fewer calls than forecast all day.",
}
ABANDON_EUR, LATE_EUR = 50.0, 8.0        # the prices of the RL reward (workforce/env.py RewardConfig)
_RL = {}


def rl_policy():
    """The trained PPO agent, or (None, why) if it cannot be loaded on this computer."""
    if "policy" not in _RL:
        try:
            from workforce.env import RLPolicy
            paths = sorted(MODELS.glob("ppo_staffing*.zip"))
            if not paths:
                raise FileNotFoundError("no trained agent in workforce/models (python -m workforce.train)")
            _RL["policy"], _RL["why"] = RLPolicy.load(paths[0]), paths[0].name
        except Exception as error:                     # RL packages not installed, or no model
            _RL["policy"], _RL["why"] = None, f"{type(error).__name__}: {error}"
    return _RL["policy"], _RL["why"]


def workforce_day(scenario: str, day: int) -> dict:
    """One simulated day of the floor, the same day for every policy, recorded every 15 minutes."""
    from workforce.manager import FixedRulePolicy, MarginRulePolicy, ReactiveRulePolicy
    from workforce.simulation import ALL_SCENARIOS, QUEUES, CallCenter, CenterConfig, Policy

    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")

    class Recorder(Policy):
        def __init__(self, policy):
            self.policy, self.log = policy, []

        def decide(self, snapshot):
            wanted = self.policy.decide(snapshot)
            self.log.append((snapshot, wanted))
            return wanted

    plan = FixedRulePolicy()
    policies = {"Erlang C plan": FixedRulePolicy(), "Reactive rule": ReactiveRulePolicy(),
                "Margin rule": MarginRulePolicy()}
    agent, why = rl_policy()
    if agent is not None:
        policies["RL agent (PPO)"] = agent

    out = {}
    for name, policy in policies.items():
        recorder = Recorder(policy)
        result = CallCenter(recorder, seed=day, scenario=scenario).run()
        k = result.kpis()
        answered = sum(result.answered.values())
        late = answered - sum(result.within_sla.values())
        total = result.staff_cost + ABANDON_EUR * sum(result.abandoned.values()) + LATE_EUR * late
        log = recorder.log
        timeline = []
        for i, (snap, wanted) in enumerate(log):
            staffed = log[i + 1][0].staffed if i + 1 < len(log) else snap.staffed   # what really happened
            timeline.append({
                "minute": snap.minute, "staffed": {q: staffed[q] for q in QUEUES},
                "waiting": {q: snap.waiting[q] for q in QUEUES},
                "plan": {q: plan.decide(snap)[q] for q in QUEUES},
                "calls": {q: snap.last_calls.get(q, 0) for q in QUEUES},
                "expected": {q: round(snap.expected_calls.get(q, 0), 1) for q in QUEUES},
                "sla": snap.last_sla, "abandoned": snap.last_abandoned, "priority": bool(wanted.get("priority")),
            })
        out[name] = {"kpis": {"calls": k["calls"], "sla": k["sla"], "abandonment_rate": k["abandonment_rate"],
                              "avg_wait_s": k["avg_wait_s"], "staff_cost_eur": round(k["staff_cost_eur"]),
                              "total_cost_eur": round(total), "occupancy": k["occupancy"]},
                     "timeline": timeline}
    return {"scenario": scenario, "day": day, "open_hour": CenterConfig().open_hour, "queues": list(QUEUES),
            "rl": {"available": agent is not None, "why": why}, "policies": out,
            "scenarios": [{"id": s, "text": SCENARIO_TEXT.get(s, "")} for s in ALL_SCENARIOS]}


def workforce() -> dict:
    out = {}
    for name in ("comparison", "twin"):
        try:
            out[name] = json.loads((MODELS / f"{name}.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            out[name] = None
    return out


# ---------------------------------------------------------------- the web server

class Handler(BaseHTTPRequestHandler):
    server_version = "CallVerseStudio/1.0"

    def log_message(self, fmt, *args):            # keep the terminal quiet, except for calls
        if "/api/call?" in (args[0] if args else ""):
            sys.stderr.write(f"[studio] {args[0]}\n")

    def send_json(self, data, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/health":
                self.send_json(health())
            elif url.path == "/api/options":
                self.send_json(options())
            elif url.path == "/api/customer":
                card = customer_card(q.get("id", "").strip())
                self.send_json(card or {"error": "Unknown customer id."}, 200 if card else 404)
            elif url.path == "/api/call":
                self.stream_call(q)
            elif url.path == "/api/calls":
                self.send_json(calls_list(q.get("run") or None))
            elif url.path.startswith("/api/calls/"):
                row = one_call(url.path.rsplit("/", 1)[1])
                self.send_json(row or {"error": "Call not found."}, 200 if row else 404)
            elif url.path == "/api/stats":
                self.send_json(stats(q.get("run") or None))
            elif url.path == "/api/workforce/day":
                self.send_json(workforce_day(q.get("scenario", "network_outage"), int(q.get("day", 10000))))
            elif url.path == "/api/workforce":
                self.send_json(workforce())
            else:
                self.send_json({"error": "Not found."}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            try:
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)
            except Exception:
                pass

    def stream_call(self, q: dict):
        from callverse.agents.client.scenarios import REASONS
        customer, reason, persona = q.get("customer", "").strip(), q.get("reason", ""), q.get("persona", "")
        if reason not in REASONS or persona not in options_personas():
            self.send_json({"error": "Choose a reason and a mood."}, 400)
            return
        call_id = f"studio-{datetime.now():%Y%m%d-%H%M%S}"
        state = CallState(call_id=call_id, customer_id=customer, persona=persona, call_reason=reason)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(event: dict):
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        send({"type": "call", "call_id": call_id, "customer": customer_card(customer), "reason": reason,
              "persona": persona})
        for event in call_events(state, save=q.get("save", "1") == "1"):
            send(event)


def options_personas() -> set[str]:
    from callverse.agents.client.personas import PERSONAS
    return set(PERSONAS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    address = f"http://127.0.0.1:{args.port}"
    print(f"CallVerse Studio is running: {address}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
