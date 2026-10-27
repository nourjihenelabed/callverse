"""Complaint specialist: fixed rules, like when to call the boss and the compensation limits.

Plain code, no AI model. The numbers come from data/policies/complaint_procedures.md
and data/policies/compensation_grid.md: if a rule changes there, change it here too.
"""

import re
import unicodedata
from datetime import datetime

from callverse.state import ComplaintCategory

# ---------------------------------------------------------------- compensation limits (euro cents)

# compensation_grid.md, "Plafond automatique selon la gravité"
AUTO_CAP_CENTS = {1: 0, 2: 500, 3: 1000, 4: 2000}          # granted without a supervisor
SUPERVISOR_CAP_CENTS = {1: 500, 2: 1500, 3: 3000, 4: 5000}  # the most a supervisor can grant

# ---------------------------------------------------------------- when a supervisor must decide

# complaint_procedures.md, "Cas où un superviseur doit décider"
REPEAT_LIMIT = 3                # the 3rd complaint of the same category in 12 months
HISTORY_DAYS = 365
CHURN_RISK_LIMIT = 0.70         # never tell the customer about this number

# "today" for the 12-month history: the same date scripts/seed_data.py uses,
# so the generated complaints are counted. Change both together.
HISTORY_REFERENCE_DATE = datetime(2026, 1, 1)

# words that mean a legal threat (compared without accents, in lower case)
LEGAL_WORDS = (
    "avocat", "proces", "plainte", "tribunal", "justice", "juridique",
    "association de consommateurs", "ufc", "que choisir", "mediateur",
)

# ---------------------------------------------------------------- steps that can never be skipped

# complaint_procedures.md, "Étapes obligatoires par catégorie"
REQUIRED_ACTIONS: dict[ComplaintCategory, str] = {
    "billing": "vérifier la facture concernée et les produits du client",
    "technical": "vérifier les pannes connues dans la zone et lancer un diagnostic à distance",
    "contract": "vérifier le contrat signé et la date de début d'engagement",
    "service_quality": "présenter des excuses claires et noter le problème",
    "termination": "recalculer les frais de résiliation selon les conditions du contrat",
    "delivery": "vérifier le suivi du colis ou du rendez-vous technicien et proposer un nouveau rendez-vous sous 72 heures",
}

ESCALATION_ACTION = "informer le client que son dossier est transmis à un responsable, qui le recontacte sous 48 heures ouvrées"


def _plain(text: str) -> str:
    """Lower case, no accents: "Procès" -> "proces"."""
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def mentions_legal_threat(text: str) -> bool:
    plain = _plain(text)
    return any(re.search(rf"\b{re.escape(word)}s?\b", plain) for word in LEGAL_WORDS)


def check_compensation(severity: int, cents: int) -> tuple[int, bool]:
    """Keeps a compensation inside the grid. Returns (allowed cents, needs a supervisor).

    Above the automatic cap, a supervisor must decide. Above the supervisor cap,
    the amount is cut down to that cap.
    """
    cents = max(0, min(cents, SUPERVISOR_CAP_CENTS[severity]))
    return cents, cents > AUTO_CAP_CENTS[severity]


# compensation_grid.md, "Coupure de service": the box price / 30, per day without service, after 48 hours
DAILY_OUTAGE_CREDIT_CENTS = {"box_fibre": 117, "box_adsl": 83}
OUTAGE_MIN_DAYS = 3                     # "plus de 48 heures"
DAY_WORDS = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10}
WEEK_WORDS = {"une": 7, "deux": 14}


def outage_days(text: str) -> int | None:
    """How many days the customer says the outage lasts: "coupé depuis 4 jours" -> 4, "une semaine" -> 7,
    "depuis ce matin" -> 0, "depuis hier" -> 1. None when the customer gives no duration."""
    plain = _plain(text)
    if re.search(r"avant-hier|avant hier", plain):
        return 2
    if re.search(r"\bhier\b", plain):
        return 1
    if re.search(r"ce matin|aujourd'?hui|ce soir|cet apres-midi|depuis (une|\d+) heures?|tout a l'heure", plain):
        return 0                                     # the same day: under 48 hours, no outage credit
    if match := re.search(r"\b(\d+)\s*jours?\b", plain):
        return int(match.group(1))
    if match := re.search(rf"\b({'|'.join(DAY_WORDS)})\s+jours?\b", plain):
        return DAY_WORDS[match.group(1)]
    if match := re.search(rf"\b({'|'.join(WEEK_WORDS)})\s+semaines?\b", plain):
        return WEEK_WORDS[match.group(1)]
    return None


def outage_credit(days: int | None, box_code: str | None) -> int | None:
    """The outage credit of the grid, or None when it does not apply (or the facts are unknown)."""
    if days is None or box_code not in DAILY_OUTAGE_CREDIT_CENTS or days < OUTAGE_MIN_DAYS:
        return None
    return days * DAILY_OUTAGE_CREDIT_CENTS[box_code]


def money_severity(text: str) -> int:
    """The lowest severity the customer's own numbers allow (complaint_procedures.md): money lost is at
    least 2, a gap of more than 50 € is 3. "75,94 € au lieu de 37,97 €" -> 2. No amount -> 1."""
    amounts = [float(a.replace(",", ".")) for a in re.findall(r"(\d+(?:,\d{1,2})?)\s*(?:€|euros?)", text)]
    if not amounts:
        return 1
    return 3 if max(amounts) - min(amounts) > 50 else 2


def tidy_compensation(category: str, cents: int) -> int:
    """Turns the model's amount into one the grid can give.

    The grid gives whole euros (5 €, 10 €...), except the outage credit of the technical
    category (1,17 € or 0,83 € per day, at least 2 days, so never under 1 €). An amount
    under 1 € is a model mistake (cents and euros mixed up): no compensation.
    """
    if cents < 100:
        return 0
    return cents if category == "technical" else round(cents / 100) * 100


def same_step(a: str, b: str) -> bool:
    """True when two actions are the same step, even with another capital letter or a few
    more words: "vérifier la facture" and "Vérifier la facture concernée avant toute proposition."."""
    key_a, key_b = (" ".join(re.findall(r"[a-z0-9]+", _plain(text))) for text in (a, b))
    return bool(key_a and key_b) and (key_a.startswith(key_b) or key_b.startswith(key_a))


def escalation_reasons(
    *, legal_threat: bool, repeat_count: int, churn_risk: float | None, over_cap: bool
) -> list[str]:
    """The supervisor cases that apply. An empty list means the specialist decides alone.

    repeat_count is the number of EARLIER complaints of the same category, so the
    current one is number repeat_count + 1.
    """
    reasons = []
    if legal_threat:
        reasons.append("menace juridique")
    if repeat_count + 1 >= REPEAT_LIMIT:
        reasons.append(f"réclamation répétée ({repeat_count + 1}e dans la même catégorie en 12 mois)")
    if churn_risk is not None and churn_risk >= CHURN_RISK_LIMIT:
        reasons.append("risque élevé de résiliation")
    if over_cap:
        reasons.append("compensation au-delà du plafond automatique")
    return reasons
