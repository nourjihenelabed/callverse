"""Advisor: the tools the AI model can use during a call.

- search_policies(question): the company rules (data/policies), through callverse.knowledge
- get_complaint_history(): the caller's past complaints
- customer_file(): the caller's offers, contract, bill and payment method. Not a model tool:
  agent.py always reads it before the model answers and puts it in the instructions.

The customer data is tied to the caller of this call: the model cannot
ask for another customer's file. They never read `churned` (the real answer,
for evaluation only) nor `churn_risk` (internal, never shown to the customer).
"""

import json
import re
from functools import lru_cache

from langchain_core.tools import StructuredTool
from sqlalchemy import select

from callverse import knowledge
from callverse.config import PROJECT_ROOT, load_settings
from callverse.db import complaints, customer_products, customers, get_engine

CONTRACTS = {
    "sans_engagement": "sans engagement",
    "engagement_12_mois": "engagement de 12 mois",
    "engagement_24_mois": "engagement de 24 mois",
}
PAYMENTS = {
    "prelevement": "prélèvement automatique",
    "carte_bancaire": "carte bancaire (paiement automatique)",
    "paiement_en_ligne": "paiement en ligne chaque mois",
    "cheque": "chèque",
}
MAX_COMPLAINTS = 5


def euros(cents: int) -> str:
    """2999 -> '29,99 €' (French format)."""
    return f"{cents / 100:.2f}".replace(".", ",") + " €"


@lru_cache
def catalogue() -> dict:
    """Product code -> (name, monthly price in cents), read from data/policies/catalogue.md."""
    path = PROJECT_ROOT / load_settings()["paths"]["policies"] / "catalogue.md"
    rows = re.findall(r"^\| (\w+) \| ([^|]+?) \| ([\d,]+) € \|", path.read_text(encoding="utf-8"), re.M)
    return {code: (name, round(float(price.replace(",", ".")) * 100)) for code, name, price in rows}


# ---------------------------------------------------------------- what each tool returns

def policy_search(question: str) -> str:
    hits = knowledge.search_policies(question)
    if not hits:
        return "Aucune règle trouvée."
    return "\n\n".join(f"[{h['source']} > {h['section']}]\n{h['text']}" for h in hits)


def customer_file(customer_id: str) -> str:
    with get_engine().connect() as conn:
        row = conn.execute(
            select(customers.c.customer_id, customers.c.tenure_months, customers.c.contract,
                   customers.c.payment_method, customers.c.paperless_billing,
                   customers.c.monthly_charge_cents, customers.c.total_charge_cents)
            .where(customers.c.customer_id == customer_id)
        ).first()
        if row is None:
            return f"Aucun client trouvé avec l'identifiant {customer_id}."
        codes = conn.execute(
            select(customer_products.c.product_code).where(customer_products.c.customer_id == customer_id)
        ).scalars().all()
    names = catalogue()
    products = [f"{names[c][0]} ({euros(names[c][1])}/mois)" if c in names else c for c in codes]
    return json.dumps({
        "identifiant": row.customer_id,
        "client_depuis_mois": row.tenure_months,
        "contrat": CONTRACTS.get(row.contract, row.contract),
        "moyen_de_paiement": PAYMENTS.get(row.payment_method, row.payment_method),
        "facture_electronique": bool(row.paperless_billing),
        "offres": products,
        "facture_mensuelle": euros(row.monthly_charge_cents),
        "total_paye_depuis_le_debut": euros(row.total_charge_cents),
    }, ensure_ascii=False)


def complaint_history(customer_id: str) -> str:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(complaints.c.opened_at, complaints.c.category, complaints.c.severity,
                   complaints.c.status, complaints.c.description, complaints.c.compensation_cents)
            .where(complaints.c.customer_id == customer_id)
            .order_by(complaints.c.opened_at.desc())
            .limit(MAX_COMPLAINTS)
        ).all()
    if not rows:
        return "Aucune réclamation passée pour ce client."
    return json.dumps([{
        "date": r.opened_at.date().isoformat() if r.opened_at else None,
        "categorie": r.category,
        "gravite": r.severity,
        "statut": "résolue" if r.status == "resolved" else "non résolue",
        "description": r.description,
        "compensation": euros(r.compensation_cents or 0),
    } for r in rows], ensure_ascii=False)


# ---------------------------------------------------------------- the tools given to the model

def build_tools(customer_id: str) -> list[StructuredTool]:
    """The tools the model can call during one call. The complaint history is bound to the caller."""
    return [
        StructuredTool.from_function(
            func=policy_search, name="search_policies",
            description="Cherche dans les règles de CallVerse Telecom (catalogue et prix, contrats, procédures, "
                        "FAQ). À utiliser avant de donner un prix, un délai ou une règle. "
                        "question : la question en français."),
        StructuredTool.from_function(
            func=lambda: complaint_history(customer_id), name="get_complaint_history",
            description="Donne les dernières réclamations du client qui appelle, avec leur statut."),
    ]
