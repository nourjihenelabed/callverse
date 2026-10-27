"""Reviewer: the checks done by plain code, with the evidence for each finding.

No AI model here: the same call always gets the same result, and each finding quotes
what the advisor said, so the grade can be explained.

- amount_findings: every euro amount the advisor said must be a known one.
- compliance_findings: the rules of data/policies/agent_guidelines.md that code can check.
- resolution_score, expected_escalation: from the outcome and the supervisor rules.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from callverse.agents.complaint import rules
from callverse.config import PROJECT_ROOT, load_settings
from callverse.state import CallState

AMOUNT = re.compile(r"(\d+(?:[,.]\d{1,2})?)\s*(?:€|euros?\b)")
SEARCH_TOOLS = {"search_policies", "get_customer", "get_complaint_history"}
CHURN_WORDS = re.compile(r"risque (élevé )?de (départ|résiliation)|churn|indicateur (interne )?de risque", re.I)
SECRET_WORDS = re.compile(r"mot de passe|num[ée]ro (complet )?de (votre )?carte|code (secret|confidentiel)", re.I)
ESCALATION_WORDS = re.compile(r"responsable|superviseur|48 ?h", re.I)
COMPENSATION_WORDS = re.compile(r"compensation|geste commercial|\bun avoir\b|dédommag|indemni", re.I)

RESOLUTION = {"resolved": 5, "escalated": 4, "refused": 2, "abandoned": 1}


@dataclass
class Finding:
    criterion: str          # "accuracy" or "compliance"
    severity: int           # how many points it costs, 1 to 4
    text: str               # French, shown in the report
    tip: str                # French, the recommendation


def cents(text: str) -> int:
    return round(float(text.replace(",", ".")) * 100)


def euros(value: int) -> str:
    return f"{value / 100:.2f}".replace(".", ",") + " €"


def advisor_lines(state: CallState) -> list[str]:
    return [m.content for m in state.messages if m.role == "advisor"]


def client_text(state: CallState) -> str:
    return " ".join(m.content for m in state.messages if m.role == "client")


# ---------------------------------------------------------------- accuracy: the amounts

@lru_cache
def policy_amounts() -> frozenset[int]:
    """Every amount written in the company rules (prices, caps, fees)."""
    folder = PROJECT_ROOT / load_settings()["paths"]["policies"]
    found = set()
    for path in folder.glob("*.md"):
        found |= {cents(a) for a in AMOUNT.findall(path.read_text(encoding="utf-8"))}
    return frozenset(found)


def known_amounts(state: CallState, facts: dict) -> set[int]:
    """Amounts the advisor may say: the rules, the customer's bill and offers, what the customer said,
    the specialist's decision, and simple sums of them (the bill plus or minus one offer)."""
    known = set(policy_amounts()) | {cents(a) for a in AMOUNT.findall(client_text(state))}
    prices = facts.get("prices", {})                     # this customer's offers: code -> cents
    bill = facts.get("bill_cents")
    known |= set(prices.values())
    if bill:
        known |= {bill, 2 * bill}
        known |= {bill + p for p in catalogue_prices()} | {bill - p for p in prices.values()}
    result = state.complaint_result
    if result is not None:
        known.add(result.compensation_cents)
    for rate in rules.DAILY_OUTAGE_CREDIT_CENTS.values():      # outage credit: days x daily rate
        known |= {days * rate for days in range(1, 31)}
    return known


def catalogue_prices() -> list[int]:
    from callverse.agents.advisor.tools import catalogue
    return [price for _, price in catalogue().values()]


def amount_findings(state: CallState, facts: dict) -> list[Finding]:
    known, findings = known_amounts(state, facts), []
    decided = state.complaint_result.compensation_cents if state.complaint_result else None
    for line in advisor_lines(state):
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            for amount in AMOUNT.findall(sentence):
                value = cents(amount)
                if decided is not None and COMPENSATION_WORDS.search(sentence) and value != decided:
                    findings.append(Finding(
                        "accuracy", 3, f"Compensation de {euros(value)} annoncée, alors que le spécialiste a décidé "
                        f"{euros(decided)} : « {sentence} »",
                        "Annoncer uniquement la compensation décidée par le spécialiste."))
                elif value not in known:
                    findings.append(Finding(
                        "accuracy", 2, f"Montant de {euros(value)} introuvable dans les règles et le dossier : "
                        f"« {sentence} »", "Ne donner que des montants trouvés dans les règles ou le dossier client."))
    return findings


# ---------------------------------------------------------------- compliance: agent_guidelines.md

def compliance_findings(state: CallState) -> list[Finding]:
    findings, said = [], " ".join(advisor_lines(state))
    used = {t.name for t in state.tools_used}

    if advisor_lines(state) and not used & SEARCH_TOOLS:
        findings.append(Finding("compliance", 2, "Aucune recherche dans les règles ni dans le dossier client "
                                "avant de répondre.", "Toujours consulter les règles et le dossier avant de répondre."))
    if match := CHURN_WORDS.search(said):
        findings.append(Finding("compliance", 4, f"L'indicateur interne de risque de départ a été mentionné "
                                f"(« {match.group(0)} »).", "Ne jamais parler au client du risque de départ."))
    if match := SECRET_WORDS.search(said):
        findings.append(Finding("compliance", 4, f"Une information secrète a été demandée (« {match.group(0)} »).",
                                "Ne jamais demander de mot de passe ni de numéro complet de carte."))
    if state.is_complaint and state.complaint_result is None:
        findings.append(Finding("compliance", 3, "Réclamation traitée sans le spécialiste des réclamations.",
                                "Demander l'aide du spécialiste avant de proposer une solution."))
    result = state.complaint_result
    if result is not None and not result.escalate and result.compensation_cents > 0:
        announced = {cents(a) for a in AMOUNT.findall(said)}
        if result.compensation_cents not in announced:
            findings.append(Finding("compliance", 2, f"Le spécialiste a décidé une compensation de "
                                    f"{euros(result.compensation_cents)}, mais le client n'en a pas été informé.",
                                    "Annoncer au client la compensation décidée par le spécialiste."))
    retention = state.retention
    if retention is not None and retention.offer_key and retention.offer_key.lower() not in said.lower():
        findings.append(Finding("compliance", 2, f"Le client voulait partir et un responsable a proposé « "
                                f"{retention.offer} », mais l'offre n'a pas été présentée.",
                                "Présenter au client l'offre de fidélisation décidée par le responsable."))
    if retention is not None and retention.supervisor and not retention.offer and not ESCALATION_WORDS.search(said):
        findings.append(Finding("compliance", 2, "Le client voulait partir et son dossier est transmis à un "
                                "responsable, mais il n'en a pas été informé.",
                                "Dire au client qu'un responsable le recontactera sous 48 heures ouvrées."))
    if result is not None and result.escalate and not ESCALATION_WORDS.search(said):
        findings.append(Finding("compliance", 3, "Le dossier est transmis à un superviseur, mais le client n'en "
                                "a pas été informé.", "Dire au client qu'un responsable le recontactera sous "
                                "48 heures ouvrées."))
    return findings


# ---------------------------------------------------------------- resolution and escalation

def resolution_score(state: CallState) -> tuple[int, str]:
    outcome = state.outcome or "abandoned"
    score = RESOLUTION[outcome]
    turns = state.turn
    if outcome == "resolved" and turns > 5:
        score -= 1
    text = {"resolved": "Le client a accepté la réponse", "escalated": "Le dossier a été transmis à un responsable",
            "refused": "Le client a refusé la solution proposée",
            "abandoned": "Le client a raccroché ou l'appel a atteint la limite de durée"}[outcome]
    return score, f"{text}, en {turns} message(s) du client."


def expected_escalation(state: CallState, churn_risk: float | None) -> tuple[bool, list[str]]:
    """The supervisor rules of complaint_procedures.md, recomputed independently from the call."""
    result = state.complaint_result
    over_cap = result.compensation_cents > rules.AUTO_CAP_CENTS[result.severity]
    reasons = rules.escalation_reasons(
        legal_threat=rules.mentions_legal_threat(client_text(state)),
        repeat_count=result.repeat_count,
        churn_risk=churn_risk,
        over_cap=over_cap,
    )
    return bool(reasons), reasons


def score_from(findings: list[Finding], criterion: str, start: int = 5) -> int:
    return max(1, start - sum(f.severity for f in findings if f.criterion == criterion))
