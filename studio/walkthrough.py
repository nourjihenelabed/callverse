"""Scenario walkthroughs for the studio: which agents, files, functions and data a call goes through.

The Scenarios tab places a real call (the same graph as the Live floor, callverse/graph.py) and shows,
step by step, every function below that runs: what it read, what it decided, and why the call moved to
the next agent. To see that, the functions are wrapped while the call runs, then put back. Nothing in
callverse/ changes, and only one call is traced at a time (another call at the same moment runs untraced).

    with tracing(events):           # events: a list that receives one dict per function that ran
        run the call
"""

import functools
import importlib
import threading
from collections import deque
from contextlib import contextmanager

# ---------------------------------------------------------------- the agents of a call

# Each step of a call: the file and the function the graph runs (callverse/graph.py build_graph),
# what it reads and what it writes into the case file (CallState).
NODES = {
    "client": {"name": "Customer", "file": "callverse/agents/client/agent.py", "func": "client_node",
               "reads": "customers, customer_products (PostgreSQL) · the mood in personas.py",
               "writes": "messages, turn, outcome, satisfaction"},
    "advisor": {"name": "Advisor", "file": "callverse/agents/advisor/agent.py", "func": "advisor_node",
                "reads": "customer file (customers) · rules search over data/policies/*.md (Chroma)",
                "writes": "intent, is_complaint, wants_to_leave, messages, tools_used"},
    "complaint": {"name": "Complaint specialist", "file": "callverse/agents/complaint/steps.py",
                  "func": "complaint_node",
                  "reads": "complaints, tickets, customers (PostgreSQL) · rules.py · rules search",
                  "writes": "complaint_result, and a ticket in PostgreSQL"},
    "retention": {"name": "Retention supervisor", "file": "callverse/agents/retention/agent.py",
                  "func": "retention_node", "reads": "customers.churn_risk (churn model), customer_products",
                  "writes": "retention"},
    "end_call": {"name": "Turn limit", "file": "callverse/graph.py", "func": "end_call",
                 "reads": "call.max_turns in settings.yaml", "writes": "outcome = abandoned"},
    "complaint_follow_up": {"name": "Ticket follow-up", "file": "callverse/agents/complaint/steps.py",
                            "func": "follow_up_node", "reads": "complaint_result, outcome",
                            "writes": "tickets.state (PostgreSQL)"},
    "quality": {"name": "Reviewer", "file": "callverse/agents/quality/agent.py", "func": "quality_node",
                "reads": "customers, catalogue · checks.py", "writes": "quality_report"},
}

# ---------------------------------------------------------------- the scenarios to choose from

SCENARIOS = [
    {"id": "invoice", "title": "Calm customer, invoice question", "reason": "check_invoice", "persona": "calm",
     "pick": "low_risk",
     "story": "The basic loop: the customer asks, the advisor reads the customer file and the rules, answers, "
              "and the reviewer grades the call.",
     "route": [("client", "opens the call: the LLM writes why they call, from the account and the mood"),
               ("advisor", "understands the request, reads the customer file, searches the rules, answers"),
               ("client", "accepts the answer: outcome resolved"),
               ("quality", "grades the call on 6 criteria")]},
    {"id": "outage", "title": "Impatient customer, internet down", "reason": "internet_outage",
     "persona": "impatient", "pick": "low_risk_box",
     "story": "The rules search uses the request's subject (outage) as well as the customer's words. An impatient "
              "customer hangs up after 4 messages without a solution (personas.py).",
     "route": [("client", "opens the call: internet down since this morning"),
               ("advisor", "searches the troubleshooting steps (procedures.md) and answers"),
               ("client", "continues, accepts, or hangs up when patience runs out"),
               ("quality", "grades the call; a hang-up gives resolution 1")]},
    {"id": "repeat_complaint", "title": "Third billing complaint", "reason": "complaint_billing",
     "persona": "dissatisfied", "pick": "repeat_billing",
     "story": "A customer who already complained about billing twice this year. The complaint specialist decides "
              "the money with rules, the 3rd complaint goes to a supervisor, and a ticket is opened.",
     "route": [("client", "complains: the last bill doubled"),
               ("advisor", "understands a complaint: is_complaint = true"),
               ("complaint", "7 steps: classify, history, rules, draft, check the cap, decide, open a ticket"),
               ("advisor", "tells the decision, the escalation and the ticket reference"),
               ("client", "accepts to wait for the manager: outcome escalated"),
               ("complaint_follow_up", "the ticket stays open for the supervisor"),
               ("quality", "grades the call and checks the escalation was right")]},
    {"id": "leaving", "title": "High churn risk, wants to cancel", "reason": "cancel_contract",
     "persona": "likely_to_churn", "pick": "demo",
     "story": "The churn model gives this customer a high risk. The retention supervisor (code only) chooses a "
              "loyalty offer from compensation_grid.md, and a guard makes sure the advisor says it.",
     "route": [("client", "wants to cancel the contract"),
               ("advisor", "understands a departure: wants_to_leave = true"),
               ("retention", "reads the churn risk (≥ 0.70) and picks the offer"),
               ("advisor", "presents the offer word for word"),
               ("client", "accepts, refuses or hangs up"),
               ("quality", "grades the call")]},
    {"id": "offer", "title": "Customer interested in an offer", "reason": "offer_information",
     "persona": "interested_in_offer", "pick": "low_risk",
     "story": "Prices come from the catalogue in the rules; a guard stops the advisor from changing a contract the "
              "customer did not ask to change.",
     "route": [("client", "asks about an offer they don't have"),
               ("advisor", "answers from the catalogue (catalogue.md) and the customer file"),
               ("client", "asks more, accepts or leaves"),
               ("quality", "checks every price said against the catalogue")]},
]


def pick_customer(kind: str, demo: str) -> str:
    """A customer of the database that fits the scenario (the demo customer if the database says nothing)."""
    from datetime import timedelta

    from sqlalchemy import func, select

    from callverse.agents.complaint import rules
    from callverse.db import complaints, customer_products, customers, get_engine

    with get_engine().connect() as conn:
        if kind == "repeat_billing":
            since = rules.HISTORY_REFERENCE_DATE - timedelta(days=rules.HISTORY_DAYS)
            row = conn.execute(
                select(complaints.c.customer_id, func.count().label("n"))
                .where(complaints.c.category == "billing", complaints.c.opened_at >= since)
                .group_by(complaints.c.customer_id).order_by(func.count().desc(), complaints.c.customer_id)
                .limit(1)).first()
            return row.customer_id if row else demo
        if kind in ("low_risk", "low_risk_box"):
            query = select(customers.c.customer_id).where(customers.c.churn_risk < 0.3)
            if kind == "low_risk_box":
                with_box = select(customer_products.c.customer_id).where(customer_products.c.product_code.like("box_%"))
                query = query.where(customers.c.customer_id.in_(with_box))
            found = conn.execute(query.order_by(customers.c.customer_id).limit(1)).scalar()
            return found or demo
    return demo


def scenario_list(demo: str) -> list[dict]:
    """The scenarios with the customer picked for each, and the route the code should take."""
    out = []
    for s in SCENARIOS:
        try:
            customer = pick_customer(s["pick"], demo)
        except Exception:                           # no database: the page still shows the routes
            customer = demo
        out.append({k: v for k, v in s.items() if k not in ("route", "pick")}
                   | {"customer": customer, "route": [{"node": n, "text": t} for n, t in s["route"]]})
    return out


# ---------------------------------------------------------------- what each function says when it runs

def _eur(cents) -> str:
    return f"{(cents or 0) / 100:.2f} €".replace(".", ",")


def _short(text, n: int = 70) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _changed(name: str, what: str):
    """For the advisor's guards: say something only when the guard changed the answer."""
    def say(args, kwargs, result):
        return f"guard {name}: {what}" if result != args[0] else None
    return say


def _route_client(args, kwargs, result):
    state = args[0]
    if result == "advisor":
        return "→ advisor: the call goes on (no outcome yet)"
    return f"→ {result}: the customer ended the call (outcome {state.outcome})"


def _route_advisor(args, kwargs, result):
    why = {"complaint": "is_complaint is true and no complaint decision yet",
           "retention": "wants_to_leave is true and no supervisor decision yet",
           "end_call": "the turn limit is reached",
           "client": "nothing to hand over: the customer's turn"}
    return f"→ {result}: {why.get(result, '')}"


def _route_review(args, kwargs, result):
    return ("→ complaint_follow_up: the complaint has a ticket to update" if result == "complaint_follow_up"
            else "→ quality: the reviewer always runs last")


def _search(args, kwargs, result):
    hits = list(result or [])
    found = ", ".join(f"{h['source']} > {h['section']}" for h in hits[:3]) or "nothing"
    return f"Chroma search \"{_short(args[0] if args else kwargs.get('question'), 50)}\" → {found}"


def _prefetch(args, kwargs, result):
    calls = result[1] if isinstance(result, tuple) and len(result) > 1 else []
    search = next((c for c in calls if getattr(c, "name", "") == "search_policies"), None)
    topic = (search.arguments.get("topic") if search else None)
    return ("before the LLM answers, code reads the customer file and searches the rules"
            + (f" (subject: {_short(topic, 60)})" if topic else ""))


def _client_facts(args, kwargs, result):
    r = result or {}
    return f"customers + customer_products → {r.get('offers', '?')}, bill {r.get('bill', '?')}, {r.get('contract', '?')}"


def _reply(args, kwargs, result):
    return f"LLM answers in the mood and decides: {getattr(result, 'decision', '?')}"


def _consistent(args, kwargs, result):
    return None if result == args[0] else f"code corrected the decision: {args[0]} → {result}"


def _understand(args, kwargs, result):
    return (f"LLM understood: intent \"{getattr(result, 'intent', '?')}\", complaint "
            f"{getattr(result, 'is_complaint', '?')}, leaving {getattr(result, 'wants_to_leave', '?')}")


def _classify(args, kwargs, result):
    case = args[1]
    return f"category {case.category}, severity {case.severity}/4" + (", legal threat" if case.legal_threat else "")


def _history(args, kwargs, result):
    h = args[1].history
    risk = f", churn risk {h.churn_risk:.2f}" if h.churn_risk is not None else ""
    return f"complaints table: {h.repeat_count} earlier {args[1].category} complaint(s) in 12 months{risk}"


def _rules(args, kwargs, result):
    case = args[1]
    labels = ", ".join(f"{h['source']} > {h['section']}" for h in case.hits[:3])
    return f"{len(case.hits)} rule passage(s)" + (f": {labels}" if labels else "")


def _draft(args, kwargs, result):
    case = args[1]
    return f"LLM drafts {len(case.actions)} action(s) and a compensation of {_eur(case.compensation_cents)}"


def _check(args, kwargs, result):
    from callverse.agents.complaint import rules
    case = args[1]
    cap = rules.AUTO_CAP_CENTS.get(case.severity, 0)
    return (f"rules.py keeps {_eur(case.compensation_cents)} (automatic cap for severity {case.severity}: "
            f"{_eur(cap)})" + (", the draft was over the cap" if case.over_cap else ""))


def _decide(args, kwargs, result):
    reasons = args[1].escalation_reasons
    return ("escalate to a supervisor: " + "; ".join(reasons)) if reasons else "no escalation: the specialist decides alone"


def _ticket(args, kwargs, result):
    return f"complaints + tickets tables: ticket {getattr(result, 'ticket_id', None) or '?'}"


def _follow(args, kwargs, result):
    ticket = args[0].complaint_result.ticket_id if args[0].complaint_result else "?"
    return f"ticket {ticket} → {result}" if result else f"ticket {ticket} stays open for the supervisor"


def _retention_facts(args, kwargs, result):
    r = result or {}
    return f"customers.churn_risk = {float(r.get('churn_risk') or 0):.2f} (churn model), products {sorted(r.get('codes') or [])}"


def _retention_decide(args, kwargs, result):
    risk = float(args[0] or 0)
    if not getattr(result, "high_risk", False):
        return f"risk {risk:.2f} < 0.70 → no offer, the advisor explains the cancellation conditions"
    return f"risk {risk:.2f} ≥ 0.70 → " + (f"offer: {result.offer}" if result.offer else "a manager calls back")


def _judge(args, kwargs, result):
    if result is None:
        return "LLM unavailable: neutral grades"
    return (f"LLM grades relevance {result.relevance.score}, accuracy {result.accuracy.score}, "
            f"empathy {result.empathy.score}")


def _findings(what):
    def say(args, kwargs, result):
        n = len(result or [])
        return f"{n} problem(s) found in the {what}" if n else f"no problem in the {what}"
    return say


# (module, function, kind, file, what it says). kind: llm = the model decides, code = rules decide,
# data = reads or writes the database or the rules index, route = graph.py picks the next agent.
PROBES = [
    ("callverse.graph", "after_client", "route", "callverse/graph.py", _route_client),
    ("callverse.graph", "after_advisor", "route", "callverse/graph.py", _route_advisor),
    ("callverse.graph", "to_review", "route", "callverse/graph.py", _route_review),

    ("callverse.agents.client.agent", "get_facts", "data", "callverse/agents/client/scenarios.py", _client_facts),
    ("callverse.agents.client.agent", "opening", "llm", "callverse/agents/client/agent.py",
     lambda a, k, r: f"LLM writes the first message ({a[2].persona} customer, {a[2].call_reason})"),
    ("callverse.agents.client.agent", "reply", "llm", "callverse/agents/client/agent.py", _reply),
    ("callverse.agents.client.agent", "consistent", "code", "callverse/agents/client/agent.py", _consistent),

    ("callverse.agents.advisor.agent", "understand", "llm", "callverse/agents/advisor/agent.py", _understand),
    ("callverse.agents.advisor.agent", "leaving", "code", "callverse/agents/advisor/agent.py",
     lambda a, k, r: "code spotted a departure in plain words (regex LEAVING)" if r else None),
    ("callverse.agents.advisor.agent", "prefetch", "code", "callverse/agents/advisor/agent.py", _prefetch),
    ("callverse.agents.advisor.agent", "customer_file", "data", "callverse/agents/advisor/tools.py",
     lambda a, k, r: "customers + customer_products → the customer file"),
    ("callverse.knowledge", "search_policies", "data", "callverse/knowledge.py", _search),
    ("callverse.agents.advisor.agent", "run_tool", "llm", "callverse/agents/advisor/agent.py",
     lambda a, k, r: f"the LLM calls the tool {a[1]}({_short(a[2], 60)})"),
    ("callverse.agents.advisor.agent", "repeats_itself", "code", "callverse/agents/advisor/agent.py",
     lambda a, k, r: "the answer repeated an earlier one: the LLM is asked again" if r else None),
    ("callverse.agents.complaint.rules", "refuses_decision", "code", "callverse/agents/complaint/rules.py",
     lambda a, k, r: "the customer refuses the decision → escalated (complaint_procedures.md)" if r else None),
    ("callverse.agents.advisor.agent", "keep_compensation", "code", "callverse/agents/advisor/agent.py",
     _changed("keep_compensation", "removed an amount the specialist did not decide")),
    ("callverse.agents.advisor.agent", "no_consent_without_request", "code", "callverse/agents/advisor/agent.py",
     _changed("no_consent_without_request", "removed a change the customer did not ask for")),
    ("callverse.agents.advisor.agent", "tidy_answer", "code", "callverse/agents/advisor/agent.py",
     _changed("tidy_answer", "shortened or cleaned the answer")),
    ("callverse.agents.advisor.agent", "keep_retention_offer", "code", "callverse/agents/advisor/agent.py",
     _changed("keep_retention_offer", "put back the supervisor's offer word for word")),
    ("callverse.agents.advisor.agent", "tell_escalation", "code", "callverse/agents/advisor/agent.py",
     _changed("tell_escalation", "added that a manager calls back within 48 h")),
    ("callverse.agents.advisor.agent", "tell_recourse", "code", "callverse/agents/advisor/agent.py",
     _changed("tell_recourse", "told the customer their recourse")),
    ("callverse.agents.advisor.agent", "give_reference", "code", "callverse/agents/advisor/agent.py",
     _changed("give_reference", "gave the ticket reference")),

    ("callverse.agents.complaint.steps", "open_ticket", "data", "callverse/agents/complaint/steps.py", _ticket),
    ("callverse.agents.complaint.steps", "follow_up", "data", "callverse/agents/complaint/steps.py", _follow),

    ("callverse.agents.retention.agent", "get_facts", "data", "callverse/agents/retention/agent.py",
     _retention_facts),
    ("callverse.agents.retention.agent", "decide", "code", "callverse/agents/retention/agent.py", _retention_decide),

    ("callverse.agents.quality.agent", "get_facts", "data", "callverse/agents/quality/agent.py",
     lambda a, k, r: "customers + catalogue → bill, prices, churn risk (for the checks)"),
    ("callverse.agents.quality.agent", "judge", "llm", "callverse/agents/quality/agent.py", _judge),
    ("callverse.agents.quality.checks", "amount_findings", "code", "callverse/agents/quality/checks.py",
     _findings("amounts said")),
    ("callverse.agents.quality.checks", "compliance_findings", "code", "callverse/agents/quality/checks.py",
     _findings("procedures")),
    ("callverse.agents.quality.agent", "review", "code", "callverse/agents/quality/agent.py",
     lambda a, k, r: f"report: overall {r.overall:.1f}/5 (LLM grades + code checks)"),

    ("callverse.results", "save_call", "data", "callverse/results.py",
     lambda a, k, r: f"calls + call_scores tables: saved in batch {a[1]}"),
]

# The complaint specialist's steps run from a list (steps.STEPS), so they are wrapped in that list.
STEP_PROBES = {
    "classify": ("mixed", _classify), "check_history": ("data", _history), "search_rules": ("data", _rules),
    "draft_fix": ("llm", _draft), "check_fix": ("code", _check), "decide": ("code", _decide),
}
STEPS_FILE = "callverse/agents/complaint/steps.py"

# ---------------------------------------------------------------- wrapping while one call runs

_LOCK = threading.Lock()
_ROUTE_DEPTH = threading.local()        # after_client calls to_review: report only the outer decision


def _wrap(original, kind: str, file: str, func: str, say, events):
    def traced(*args, **kwargs):
        outer = kind != "route" or getattr(_ROUTE_DEPTH, "n", 0) == 0
        if kind == "route":
            _ROUTE_DEPTH.n = getattr(_ROUTE_DEPTH, "n", 0) + 1
        try:
            result = original(*args, **kwargs)
        finally:
            if kind == "route":
                _ROUTE_DEPTH.n -= 1
        if outer:
            try:
                text = say(args, kwargs, result)
            except Exception:                   # a summary must never break the call
                text = f"{func}() ran"
            if text:
                events.append({"type": "trace", "kind": kind, "file": file, "func": func, "text": text})
        return result
    # same name and signature as the original: LangGraph reads a router's signature to call it
    functools.update_wrapper(traced, original)
    return traced


@contextmanager
def tracing(events):
    """Wraps the functions of PROBES and STEP_PROBES while the block runs. Yields False (and traces
    nothing) when another call is already being traced."""
    if not _LOCK.acquire(blocking=False):
        yield False
        return
    restore = []
    steps_module, old_steps = None, None
    try:
        for module_name, attr, kind, file, say in PROBES:
            module = importlib.import_module(module_name)
            original = getattr(module, attr)
            restore.append((module, attr, original))
            setattr(module, attr, _wrap(original, kind, file, attr, say, events))
        steps_module = importlib.import_module("callverse.agents.complaint.steps")
        old_steps = list(steps_module.STEPS)
        steps_module.STEPS[:] = [
            _wrap(step, STEP_PROBES[step.__name__][0], STEPS_FILE, step.__name__, STEP_PROBES[step.__name__][1], events)
            if step.__name__ in STEP_PROBES else step for step in old_steps]
        yield True
    finally:
        for module, attr, original in reversed(restore):
            setattr(module, attr, original)
        if steps_module is not None and old_steps is not None:
            steps_module.STEPS[:] = old_steps
        _LOCK.release()


def traced_events(call_events, state, save: bool = True):
    """The events of call_events (studio/server.py), with a "trace" event for every wrapped function that
    ran, placed before the update of the agent that ran it."""
    pending = deque()
    with tracing(pending) as on:
        if not on:
            yield {"type": "notice", "message": "Another traced call is running: this one shows the agents only."}
        for event in call_events(state, save=save):
            while pending:
                yield pending.popleft()
            yield event
        while pending:
            yield pending.popleft()
