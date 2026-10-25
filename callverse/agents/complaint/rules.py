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
