"""The case file that travels through a call, and what each worker sends and returns.

Shared by everyone. Ask the team before changing it.

How it works:
- The simulation creates a CallState when a customer calls.
- Each worker reads the CallState and returns only the fields it changes.
- `messages` and `tools_used` are added to, never replaced: a worker returns
  only its new items and LangGraph appends them.
"""

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- shared words

Persona = Literal[
    "calm",
    "impatient",
    "dissatisfied",
    "demanding",
    "interested_in_offer",
    "likely_to_churn",
]

Outcome = Literal[
    "resolved",    # the customer accepts the answer
    "refused",     # the customer refuses the solution
    "abandoned",   # the customer hangs up, or the call hit the turn limit
    "escalated",   # the case went to a supervisor
]

ComplaintCategory = Literal[
    "billing",
    "technical",
    "contract",
    "termination",
    "service_quality",
    "delivery",
]

QUALITY_CRITERIA = (
    "relevance",
    "accuracy",
    "compliance",
    "empathy",
    "resolution",
    "satisfaction",
)


# ---------------------------------------------------------------- small pieces

class Message(BaseModel):
    """One line said during the call."""

    role: Literal["client", "advisor"]
    content: str


class ToolCall(BaseModel):
    """One tool the advisor used, kept so the reviewer can check it."""

    name: str                                   # e.g. "search_policies", "get_customer"
    arguments: dict = Field(default_factory=dict)


# ---------------------------------------------------------------- complaint specialist -> advisor

class ComplaintResult(BaseModel):
    """What the complaint specialist hands back to the advisor."""

    category: ComplaintCategory
    severity: int = Field(ge=1, le=4)           # 1 = minor, 4 = critical
    repeat_count: int = Field(default=0, ge=0)  # earlier complaints of the same category
    actions: list[str]                          # what the advisor should do or offer
    compensation_cents: int = Field(default=0, ge=0)  # euros in cents: 2999 = 29.99 EUR
    rules_cited: list[str] = Field(default_factory=list)  # policy sections that justify the fix
    escalate: bool = False                      # True = a supervisor must decide
    escalation_reasons: list[str] = Field(default_factory=list)
    ticket_id: str | None = None


# ---------------------------------------------------------------- supervisor (retention) -> advisor

class RetentionResult(BaseModel):
    """What the supervisor decides when a customer wants to leave (agent_guidelines.md, compensation_grid.md)."""

    high_risk: bool                             # churn risk at or above the supervisor threshold
    supervisor: bool                            # True = a supervisor handles the case (high risk)
    offer: str | None = None                    # the loyalty offer, in French; None = no offer
    offer_key: str | None = None                # words the advisor must say to present it, e.g. "3 mois offerts"
    rules_cited: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- reviewer output

class CriterionScore(BaseModel):
    """The grade for one criterion, with the reason (explainability)."""

    score: int = Field(ge=1, le=5)
    explanation: str


class QualityReport(BaseModel):
    """What the reviewer writes after the call."""

    scores: dict[str, CriterionScore]           # one entry per name in QUALITY_CRITERIA
    overall: float = Field(ge=1, le=5)
    recommendations: list[str] = Field(default_factory=list)
    escalation_correct: bool | None = None      # complaints only: was the escalation choice right?


# ---------------------------------------------------------------- the case file

class CallState(BaseModel):
    """Everything known about one call. Who fills each field is noted on the right."""

    # set by the simulation when the call starts
    call_id: str
    customer_id: str                            # key of the customers table (IBM Telco id)
    persona: Persona
    call_reason: str                            # why the customer calls, e.g. "check_invoice", "complaint_billing"

    # the conversation
    messages: Annotated[list[Message], operator.add] = Field(default_factory=list)  # client + advisor
    turn: int = 0                               # client messages so far (the client adds 1)

    # advisor
    intent: str | None = None                   # what the advisor understood the customer wants
    is_complaint: bool = False                  # True sends the call to the complaint specialist
    wants_to_leave: bool = False                # True sends the call to the supervisor (retention) once
    tools_used: Annotated[list[ToolCall], operator.add] = Field(default_factory=list)

    # complaint specialist
    complaint_result: ComplaintResult | None = None

    # supervisor, when the customer wants to leave
    retention: RetentionResult | None = None

    # client, at the end of the call
    outcome: Outcome | None = None
    satisfaction: int | None = Field(default=None, ge=1, le=5)

    # reviewer, after the call
    quality_report: QualityReport | None = None
