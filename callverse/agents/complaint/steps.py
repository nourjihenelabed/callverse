"""Complaint specialist: the steps it follows, in order.

    1. Classify ............ asks the AI model (llama3.1)
    2. Check history ....... asks the database
    3. Search rules ........ asks the rules search
    4. Draft the fix ....... asks the AI model (llama3.1)
    5. Check the fix ....... plain code, no AI
    6. Decide / escalate ... plain code, no AI
    7. Open ticket ......... writes to the database

After the call, follow_up_node updates the ticket with the customer's answer
(graph.py runs it once the client has set an outcome).

The AI model only suggests (steps 1 and 4). The money limits and the supervisor
cases are checked by plain code in rules.py, so a model cannot talk its way past them.
If the model fails or gives a bad answer, the step falls back to a safe rule-based answer.

Input: the CallState. Output: {"complaint_result": ComplaintResult(...)}
follow_up_node returns {} : it only writes to the database.

The database, the rules search and the model are reached through a Tools object,
so tests can swap them for fakes (see tests/conftest.py).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from pydantic import BaseModel, Field

from callverse.agents.complaint import prompts, rules
from callverse.state import CallState, ComplaintCategory, ComplaintResult

log = logging.getLogger(__name__)

# the customer's answer -> the new ticket state (complaint_procedures.md, "Suivi des tickets").
# Not listed (abandoned, escalated): the ticket stays "solution_proposed".
TICKET_STATE_AFTER_CALL = {"resolved": "accepted", "refused": "refused"}

REOPEN_DAYS = 30                # same problem within 30 days of a resolution: reopen the old ticket
MAX_DRAFT_ACTIONS = 3            # the model's own actions; required steps are added after

# the rule sections the plain-code checks rely on, cited when they apply
RULE_ESCALATION = "complaint_procedures.md > Cas où un superviseur doit décider"
RULE_CAPS = "compensation_grid.md > Plafond automatique selon la gravité"
RULE_REQUIRED = "complaint_procedures.md > Étapes obligatoires par catégorie"


# ---------------------------------------------------------------- what the steps work with

@dataclass
class History:
    """What the database knows about the customer, for this category."""

    repeat_count: int = 0                       # earlier complaints of the same category, last 12 months
    churn_risk: float | None = None
    reopen_ticket_id: str | None = None         # a ticket resolved less than 30 days ago, same category


@dataclass
class Tools:
    """The outside world of the specialist. Tests replace these with fakes."""

    llm: object | None                          # None = no AI model, use the rule-based answers
    get_history: Callable[[str, ComplaintCategory], History]
    search: Callable[[str], list[dict]]
    open_ticket: Callable[[CallState, ComplaintResult, History], str]
    update_ticket: Callable[[str, str], None]   # (ticket_id, new state)


@dataclass
class Case:
    """The specialist's working notes, filled step by step."""

    category: ComplaintCategory = "billing"
    severity: int = 2
    legal_threat: bool = False
    history: History = field(default_factory=History)
    hits: list[dict] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    compensation_cents: int = 0
    rules_cited: list[str] = field(default_factory=list)
    over_cap: bool = False
    escalation_reasons: list[str] = field(default_factory=list)


class _Classification(BaseModel):
    category: ComplaintCategory
    severity: int = Field(ge=1, le=4)
    legal_threat: bool = False


class _Draft(BaseModel):
    actions: list[str]
    compensation_cents: int = Field(default=0, ge=0)
    rules_used: list[int] = Field(default_factory=list)


def _conversation(state: CallState) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in state.messages) or "(pas encore de message)"


def _client_text(state: CallState) -> str:
    return " ".join(m.content for m in state.messages if m.role == "client")


def _label(hit: dict) -> str:
    return f"{hit['source']} > {hit['section']}"


# ---------------------------------------------------------------- the 7 steps

def classify(state: CallState, case: Case, tools: Tools) -> None:
    """1. Pick the category and the severity."""
    # fallback: the call reason often says it, e.g. "complaint_billing"
    guess = state.call_reason.removeprefix("complaint_")
    if guess in rules.REQUIRED_ACTIONS:
        case.category = guess

    if tools.llm is not None:
        try:
            prompt = prompts.CLASSIFY.format(call_reason=state.call_reason, conversation=_conversation(state))
            answer = tools.llm.with_structured_output(_Classification).invoke(prompt)
            case.category, case.severity, case.legal_threat = answer.category, answer.severity, answer.legal_threat
        except Exception as error:
            log.warning("classify: model failed, using the call reason instead (%s)", error)

    # never trust the model alone on a legal threat
    case.legal_threat = case.legal_threat or rules.mentions_legal_threat(_client_text(state))


def check_history(state: CallState, case: Case, tools: Tools) -> None:
    """2. Count the customer's earlier complaints of the same category."""
    case.history = tools.get_history(state.customer_id, case.category)


def search_rules(state: CallState, case: Case, tools: Tools) -> None:
    """3. Find the company rules for this kind of complaint."""
    topic = prompts.SEARCH_TOPICS[case.category]
    questions = [q.format(topic=topic) for q in prompts.SEARCH_QUESTIONS]
    questions.append(_client_text(state) or state.call_reason)
    seen = set()
    for question in questions:
        for hit in tools.search(question):
            if _label(hit) not in seen:
                seen.add(_label(hit))
                case.hits.append(hit)


def draft_fix(state: CallState, case: Case, tools: Tools) -> None:
    """4. Suggest actions, a compensation and the rules that justify them."""
    if tools.llm is not None:
        try:
            numbered = "\n\n".join(f"[{n}] {hit['text']}" for n, hit in enumerate(case.hits, 1))
            prompt = prompts.DRAFT.format(
                category=case.category,
                severity=case.severity,
                repeat_count=case.history.repeat_count,
                conversation=_conversation(state),
                rules=numbered or "(aucune règle trouvée)",
            )
            answer = tools.llm.with_structured_output(_Draft).invoke(prompt)
            case.actions = [a.strip() for a in answer.actions if a.strip()][:MAX_DRAFT_ACTIONS]
            case.compensation_cents = answer.compensation_cents
            case.rules_cited = [_label(case.hits[n - 1]) for n in answer.rules_used if 1 <= n <= len(case.hits)]
            return
        except Exception as error:
            log.warning("draft_fix: model failed, using the safe answer instead (%s)", error)

    # safe answer without a model: the required steps, no compensation
    case.actions, case.compensation_cents = [], 0
    case.rules_cited = [_label(case.hits[0])] if case.hits else []


def check_fix(state: CallState, case: Case, tools: Tools) -> None:
    """5. Keep the compensation inside the grid and add any required step that is missing."""
    case.compensation_cents, case.over_cap = rules.check_compensation(case.severity, case.compensation_cents)

    required = rules.REQUIRED_ACTIONS[case.category]
    same = [a for a in case.actions if a.casefold() == required.casefold()]   # "Vérifier..." is "vérifier..."
    if not same:
        case.rules_cited.append(RULE_REQUIRED)
    case.actions = [required] + [a for a in case.actions if a not in same]
    if case.compensation_cents > 0 and not any(r.startswith("compensation_grid.md") for r in case.rules_cited):
        case.rules_cited.append(RULE_CAPS)


def decide(state: CallState, case: Case, tools: Tools) -> None:
    """6. Handle it here, or send it to a supervisor."""
    case.escalation_reasons = rules.escalation_reasons(
        legal_threat=case.legal_threat,
        repeat_count=case.history.repeat_count,
        churn_risk=case.history.churn_risk,
        over_cap=case.over_cap,
    )
    if case.escalation_reasons:
        case.actions.append(rules.ESCALATION_ACTION)
        case.rules_cited.append(RULE_ESCALATION)


def open_ticket(state: CallState, case: Case, tools: Tools) -> ComplaintResult:
    """7. Record the complaint and its ticket, and build the answer for the advisor."""
    result = ComplaintResult(
        category=case.category,
        severity=case.severity,
        repeat_count=case.history.repeat_count,
        actions=list(dict.fromkeys(case.actions)),
        compensation_cents=case.compensation_cents,
        rules_cited=list(dict.fromkeys(case.rules_cited)),
        escalate=bool(case.escalation_reasons),
        escalation_reasons=case.escalation_reasons,
    )
    result.ticket_id = tools.open_ticket(state, result, case.history)
    return result


STEPS = [classify, check_history, search_rules, draft_fix, check_fix, decide]


def handle_complaint(state: CallState, tools: Tools) -> ComplaintResult:
    case = Case()
    for step in STEPS:
        step(state, case, tools)
    return open_ticket(state, case, tools)


def complaint_node(state: CallState) -> dict:
    return {"complaint_result": handle_complaint(state, make_tools())}


def follow_up(state: CallState, tools: Tools) -> str | None:
    """After the call: record the customer's answer on the ticket. Returns the new state, or None."""
    result = state.complaint_result
    if result is None or result.ticket_id is None:
        return None
    # an escalated case waits for the supervisor, even if the customer said "d'accord"
    if result.escalate:
        return None
    new_state = TICKET_STATE_AFTER_CALL.get(state.outcome)
    if new_state:
        tools.update_ticket(result.ticket_id, new_state)
    return new_state


def follow_up_node(state: CallState) -> dict:
    follow_up(state, make_tools())
    return {}


# ---------------------------------------------------------------- the real tools

def make_tools() -> Tools:
    """The real model, database and rules search."""
    from callverse.knowledge import search_policies
    from callverse.llm import get_llm

    return Tools(llm=get_llm(temperature=0), get_history=db_history, search=search_policies,
                 open_ticket=db_open_ticket, update_ticket=db_update_ticket)


def db_history(customer_id: str, category: ComplaintCategory) -> History:
    from sqlalchemy import and_, func, select

    from callverse.db import complaints, customers, get_engine, tickets

    today = rules.HISTORY_REFERENCE_DATE
    since = today - timedelta(days=rules.HISTORY_DAYS)
    same = and_(complaints.c.customer_id == customer_id, complaints.c.category == category)

    with get_engine().connect() as conn:
        churn_risk = conn.execute(
            select(customers.c.churn_risk).where(customers.c.customer_id == customer_id)
        ).scalar()
        repeat_count = conn.execute(
            select(func.count()).select_from(complaints).where(same, complaints.c.opened_at >= since)
        ).scalar()
        reopen = conn.execute(
            select(tickets.c.ticket_id)
            .join(complaints, tickets.c.complaint_id == complaints.c.complaint_id)
            .where(same, tickets.c.state == "resolved",
                   complaints.c.closed_at >= today - timedelta(days=REOPEN_DAYS))
            .order_by(complaints.c.closed_at.desc())
        ).scalar()

    return History(repeat_count=repeat_count or 0, churn_risk=churn_risk, reopen_ticket_id=reopen)


def db_open_ticket(state: CallState, result: ComplaintResult, history: History) -> str:
    from sqlalchemy import func, insert, select, update

    from callverse.db import complaints, get_engine, tickets

    now = datetime.now()
    with get_engine().begin() as conn:
        # the same problem came back soon after a resolution: reopen, do not create a new one
        if history.reopen_ticket_id:
            conn.execute(
                update(tickets).where(tickets.c.ticket_id == history.reopen_ticket_id)
                .values(state="reopened", updated_at=now)
            )
            return history.reopen_ticket_id

        last = conn.execute(select(func.max(complaints.c.complaint_id))).scalar()
        number = int(last.split("-")[1]) + 1 if last else 1
        complaint_id, ticket_id = f"CMP-{number:06d}", f"TCK-{number:06d}"

        conn.execute(insert(complaints).values(
            complaint_id=complaint_id,
            customer_id=state.customer_id,
            category=result.category,
            severity=result.severity,
            description=_client_text(state)[:2000] or state.call_reason,
            status="unresolved",
            compensation_cents=result.compensation_cents,
            opened_at=now,
            closed_at=None,
        ))
        conn.execute(insert(tickets).values(
            ticket_id=ticket_id,
            complaint_id=complaint_id,
            customer_id=state.customer_id,
            state="solution_proposed",
            created_at=now,
            updated_at=now,
        ))
    return ticket_id


def db_update_ticket(ticket_id: str, new_state: str) -> None:
    from sqlalchemy import update

    from callverse.db import get_engine, tickets

    with get_engine().begin() as conn:
        conn.execute(update(tickets).where(tickets.c.ticket_id == ticket_id)
                     .values(state=new_state, updated_at=datetime.now()))
