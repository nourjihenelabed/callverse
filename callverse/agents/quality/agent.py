"""Reviewer: grades each call after it ends, explains every grade and gives tips.

Six criteria (state.QUALITY_CRITERIA), each with a score from 1 to 5 and an explanation:
- relevance, empathy: the AI model reads the call (temperature 0).
- accuracy: the AI model, then code checks every euro amount the advisor said (checks.py);
  an amount found nowhere in the rules or the customer's file lowers the grade.
- compliance: code checks the rules of agent_guidelines.md (search before answering, no churn risk,
  no secret asked, complaint sent to the specialist, escalation announced).
- resolution: from the outcome of the call.
- satisfaction: the customer's own score.
escalation_correct (complaints only): the supervisor rules recomputed independently.

Input: the finished CallState. Output: {"quality_report": QualityReport(...)}
The AI model comes from get_model() and the account from get_facts(), so tests can replace them.
"""

import logging
import re
import statistics
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select

from callverse.agents.quality import checks, prompts
from callverse.llm import get_llm
from callverse.state import QUALITY_CRITERIA, CallState, CriterionScore, QualityReport

log = logging.getLogger(__name__)

MAX_RECOMMENDATIONS = 4


class Grade(BaseModel):
    score: int = Field(ge=1, le=5)
    explanation: str


class AccuracyGrade(Grade):
    quote: str = Field(default="", description="la phrase fausse du conseiller, recopiée exactement")
    contradicts: Literal["catalogue", "dossier", "décision", "aucune"] = "aucune"


class Judgement(BaseModel):
    relevance: Grade
    accuracy: AccuracyGrade
    empathy: Grade
    recommendations: list[str] = Field(default_factory=list)


def get_model():
    """The AI model, at temperature 0 so the same call gets the same grades. Tests replace it."""
    return get_llm(temperature=0)


def get_facts(customer_id: str) -> dict:
    """The customer's account, read for the review (the reviewer may see churn_risk: it is internal)."""
    from callverse.agents.advisor.tools import catalogue, customer_file
    from callverse.db import customer_products, customers, get_engine

    with get_engine().connect() as conn:
        row = conn.execute(select(customers.c.monthly_charge_cents, customers.c.churn_risk)
                           .where(customers.c.customer_id == customer_id)).first()
        codes = list(conn.execute(select(customer_products.c.product_code)
                                  .where(customer_products.c.customer_id == customer_id)).scalars())
    names = catalogue()
    return {
        "summary": customer_file(customer_id),
        "bill_cents": row.monthly_charge_cents if row else None,
        "churn_risk": row.churn_risk if row else None,
        "prices": {c: names[c][1] for c in codes if c in names},
    }


def quality_node(state: CallState) -> dict:
    try:
        facts = get_facts(state.customer_id)
    except Exception as error:          # no database: the review goes on with what the call shows
        log.warning("reviewer: account not found (%s)", error)
        facts = {"summary": "indisponible", "bill_cents": None, "churn_risk": None, "prices": {}}
    return {"quality_report": review(state, facts, judge(state, facts))}


# ---------------------------------------------------------------- the AI model's part

def conversation(state: CallState) -> str:
    who = {"client": "Client", "advisor": "Conseiller"}
    return "\n".join(f"{who[m.role]} : {m.content}" for m in state.messages)


def decision_text(state: CallState) -> str:
    r = state.complaint_result
    if r is None:
        return "aucune (pas de réclamation)"
    return (f"catégorie {r.category}, gravité {r.severity}, actions : {'; '.join(r.actions) or 'aucune'}, "
            f"compensation {checks.euros(r.compensation_cents)}, transmis à un superviseur : "
            f"{'oui' if r.escalate else 'non'}")


def judge(state: CallState, facts: dict) -> Judgement | None:
    from callverse.agents.advisor.agent import catalogue_text

    prompt = prompts.JUDGE.format(
        tone_rule=prompts.TONE_RULES[state.persona], call_reason=state.call_reason, persona=state.persona,
        customer=facts["summary"], catalogue=catalogue_text().replace("\n", " ; "),
        decision=decision_text(state), conversation=conversation(state))
    try:
        answer = get_model().with_structured_output(Judgement).invoke(prompt)
        return answer if isinstance(answer, Judgement) else Judgement.model_validate(answer)
    except Exception as error:
        log.warning("reviewer: model failed, neutral grades (%s)", error)
        return None


# ---------------------------------------------------------------- the report

def review(state: CallState, facts: dict, judged: Judgement | None) -> QualityReport:
    neutral = Grade(score=3, explanation=prompts.NO_JUDGE)
    findings = checks.amount_findings(state, facts) + checks.compliance_findings(state)

    model_score, model_text = grounded_accuracy(judged.accuracy if judged else None, state)
    accuracy = CriterionScore(
        score=min(model_score, checks.score_from(findings, "accuracy")),
        explanation=explain(model_text, findings, "accuracy"))

    compliance_score = checks.score_from(findings, "compliance")
    compliance = CriterionScore(
        score=compliance_score,
        explanation=explain("", findings, "compliance", ok="Toutes les règles vérifiées ont été respectées."))

    resolution_score, resolution_text = checks.resolution_score(state)
    satisfaction = state.satisfaction or 1

    scores = {
        "relevance": labelled(judged.relevance if judged else neutral),
        "accuracy": accuracy,
        "compliance": compliance,
        "empathy": labelled(judged.empathy if judged else neutral),
        "resolution": CriterionScore(score=resolution_score, explanation=resolution_text),
        "satisfaction": CriterionScore(score=satisfaction, explanation=f"Note donnée par le client : {satisfaction}/5."),
    }
    assert set(scores) == set(QUALITY_CRITERIA)

    tips = [f.tip for f in findings] + (judged.recommendations if judged else [])
    tips = list(dict.fromkeys(t.strip() for t in tips if t and t.strip()))[:MAX_RECOMMENDATIONS]

    escalation_correct = None
    if state.complaint_result is not None:
        expected, _ = checks.expected_escalation(state, facts.get("churn_risk"))
        escalation_correct = expected == state.complaint_result.escalate

    return QualityReport(
        scores=scores,
        overall=round(statistics.mean(s.score for s in scores.values()), 2),
        recommendations=tips,
        escalation_correct=escalation_correct,
    )


def labelled(grade: Grade) -> CriterionScore:
    text = grade.explanation if grade.explanation == prompts.NO_JUDGE else prompts.MODEL + grade.explanation
    return CriterionScore(score=grade.score, explanation=text)


def grounded_accuracy(grade, state: CallState) -> tuple[int, str]:
    """The judge may lower accuracy only with evidence: the advisor's exact sentence and the source it
    contradicts. Without it, the criticism is not kept (the small judge once invented an error)."""
    if grade is None:
        return 3, prompts.NO_JUDGE
    text = prompts.MODEL + grade.explanation
    if grade.score >= 4:
        return grade.score, text
    if grade.contradicts == "aucune":
        why = prompts.NO_SOURCE
    elif grade.contradicts == "décision" and state.complaint_result is None:
        why = prompts.NO_DECISION
    elif not quoted(grade.quote, state):
        why = prompts.NO_QUOTE
    else:
        return grade.score, text
    return 4, f"{text} {prompts.NOT_KEPT.format(why=why, score=grade.score)}"


def quoted(quote: str, state: CallState) -> bool:
    """True when the quote really is (most of) something the advisor said."""
    words = lambda text: re.findall(r"\w+", text.lower())  # noqa: E731
    wanted = " ".join(words(quote))
    if len(wanted) < 10:
        return False
    return any(wanted in " ".join(words(line)) for line in checks.advisor_lines(state))


def explain(model_text: str, findings: list, criterion: str, ok: str = "") -> str:
    """The model's view and the code's findings, each labelled, so the reader knows who said what."""
    parts = [model_text] if model_text else []
    found = [f.text for f in findings if f.criterion == criterion]
    if found:
        parts.append(prompts.CODE + " ".join(found))
    elif ok:
        parts.append(prompts.CODE + ok)
    return " ".join(parts) or "Rien à signaler."
