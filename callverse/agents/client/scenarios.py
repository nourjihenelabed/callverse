"""Pretend customer: why people call, and how a random call is drawn.

- REASONS: each call reason, its queue (as in the workforce simulation) and a short brief
  that tells the pretend customer their situation, filled with their real account facts.
- customer_facts(): what the customer knows about their own account (from the database).
- random_call(): draws a customer, a reason and a mood for one call.

The simulation may read churn_risk here: it plays the real customer, who knows whether
they want to leave. The advisor never sees it.
"""

import random
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import select

from callverse.db import customer_products, customers, get_engine
from callverse.state import CallState, Persona

# the same shares as workforce/simulation.py (CenterConfig.call_mix)
QUEUE_SHARE = {"general": 0.55, "technical": 0.30, "complaint": 0.15}


@dataclass(frozen=True)
class Reason:
    queue: str
    brief: str              # French, filled with customer_facts()
    needs: str = ""         # "mobile" or "box": only for customers who have one
    weight: float = 1.0     # share inside its queue


REASONS: dict[str, Reason] = {
    # ---- general
    "check_invoice": Reason("general", "Vous voulez comprendre votre facture : vous payez {bill} par mois et vous "
                            "voulez savoir à quoi correspond ce montant.", weight=3),
    "offer_information": Reason("general", "Vous voulez connaître le prix et le contenu d'une offre que vous n'avez "
                                "pas encore : {other_offer}.", weight=2),
    "change_payment_method": Reason("general", "Vous payez par {payment} et vous voulez passer au prélèvement "
                                    "automatique. Vous demandez comment faire."),
    "add_option": Reason("general", "Vous voulez ajouter l'option {other_option} et vous demandez son prix et "
                         "si elle a un engagement.", needs="box"),
    "lost_sim": Reason("general", "Vous avez perdu votre carte SIM hier. Vous voulez savoir quoi faire et "
                       "combien cela coûte.", needs="mobile"),
    "rio_code": Reason("general", "Vous voulez votre code RIO pour garder votre numéro mobile chez un autre "
                       "opérateur.", needs="mobile"),
    "cancel_contract": Reason("general", "Vous voulez savoir comment résilier votre abonnement ({contract}) et "
                              "s'il y a des frais."),
    # ---- technical
    "internet_outage": Reason("technical", "Votre internet ({box}) ne marche plus depuis ce matin. Les voyants de "
                              "la box sont rouges.", needs="box", weight=3),
    "slow_internet": Reason("technical", "Votre internet ({box}) est très lent le soir depuis une semaine.",
                            needs="box", weight=2),
    "activate_sim": Reason("technical", "Vous avez reçu une nouvelle carte SIM et vous ne savez pas l'activer.",
                           needs="mobile"),
    # ---- complaints
    "complaint_billing": Reason("complaint", "Vous appelez pour faire une réclamation : votre dernière facture est de "
                                "{double_bill} au lieu de {bill} habituellement, et vous n'avez rien changé.",
                                weight=3),
    "complaint_technical": Reason("complaint", "Vous appelez pour faire une réclamation : votre internet ({box}) est "
                                  "coupé depuis 4 jours et personne ne règle le problème.", needs="box", weight=2),
    "complaint_service_quality": Reason("complaint", "Vous appelez pour faire une réclamation : la semaine dernière, on "
                                        "vous a promis un rappel qui n'a jamais eu lieu, après 40 minutes d'attente."),
    "complaint_delivery": Reason("complaint", "Vous appelez pour faire une réclamation : le technicien n'est pas venu "
                                 "au rendez-vous d'installation hier, sans prévenir.", needs="box"),
    "complaint_termination": Reason("complaint", "Vous appelez pour faire une réclamation : vous contestez les frais "
                                    "de résiliation annoncés, que vous trouvez trop élevés."),
    "complaint_contract": Reason("complaint", "Vous appelez pour faire une réclamation : le changement d'offre "
                                 "demandé le mois dernier n'a jamais été appliqué."),
}

# how often each mood calls (likely_to_churn is drawn from the customer's churn risk instead)
MOOD_WEIGHTS = {
    "general": {"calm": 5, "impatient": 2, "dissatisfied": 1, "demanding": 1, "interested_in_offer": 2},
    "technical": {"calm": 3, "impatient": 3, "dissatisfied": 2, "demanding": 1, "interested_in_offer": 0},
    "complaint": {"calm": 1, "impatient": 2, "dissatisfied": 4, "demanding": 3, "interested_in_offer": 0},
}

CONTRACTS = {"sans_engagement": "sans engagement", "engagement_12_mois": "engagement de 12 mois",
             "engagement_24_mois": "engagement de 24 mois"}
PAYMENTS = {"prelevement": "prélèvement automatique", "carte_bancaire": "carte bancaire",
            "paiement_en_ligne": "paiement en ligne", "cheque": "chèque"}


def euros(cents: int) -> str:
    return f"{cents / 100:.2f}".replace(".", ",") + " €"


@lru_cache
def _catalogue() -> dict:
    from callverse.agents.advisor.tools import catalogue
    return catalogue()


# ---------------------------------------------------------------- the customer's own account

def customer_facts(customer_id: str) -> dict:
    """What the customer knows: their offers, bill, contract and payment method. Also churn_risk,
    used only to draw the mood (the customer knows if they want to leave)."""
    with get_engine().connect() as conn:
        row = conn.execute(select(customers).where(customers.c.customer_id == customer_id)).first()
        if row is None:
            raise ValueError(f"unknown customer {customer_id}")
        codes = list(conn.execute(select(customer_products.c.product_code)
                                  .where(customer_products.c.customer_id == customer_id)).scalars())
    return facts_from(row._mapping, codes)


def facts_from(row, codes: list[str]) -> dict:
    names = {code: name for code, (name, _) in _catalogue().items()}
    others = [names[c] for c in names if c not in codes]
    return {
        "codes": codes,
        "offers": ", ".join(names.get(c, c) for c in codes) or "aucune offre",
        "bill": euros(row["monthly_charge_cents"]),
        "double_bill": euros(2 * row["monthly_charge_cents"]),
        "box": next((names[c] for c in codes if c.startswith("box_")), "aucune box"),
        "contract": CONTRACTS.get(row["contract"], row["contract"]),
        "payment": PAYMENTS.get(row["payment_method"], row["payment_method"]),
        "tenure": row["tenure_months"],
        "other_offer": next((o for o in others if not o.startswith(("Sécurité", "Sauvegarde"))), "la Box Fibre"),
        "other_option": next((names[c] for c in names if c.startswith("option_") and c not in codes),
                             "Cinéma & Séries"),
        "churn_risk": row.get("churn_risk") or 0.0,
    }


def has(facts: dict, need: str) -> bool:
    if need == "mobile":
        return "forfait_mobile" in facts["codes"]
    if need == "box":
        return any(c.startswith("box_") for c in facts["codes"])
    return True


def brief(call_reason: str, facts: dict) -> str:
    reason = REASONS.get(call_reason)
    if reason is None:      # a reason written by hand, e.g. in a test
        return f"Vous appelez pour : {call_reason.replace('_', ' ')}."
    return reason.brief.format(**facts)


# ---------------------------------------------------------------- one random call

def pick_reason(rng: random.Random, facts: dict, queue: str | None = None) -> str:
    queue = queue or rng.choices(list(QUEUE_SHARE), weights=list(QUEUE_SHARE.values()))[0]
    possible = {name: r for name, r in REASONS.items() if r.queue == queue and has(facts, r.needs)}
    return rng.choices(list(possible), weights=[r.weight for r in possible.values()])[0]


def pick_persona(rng: random.Random, queue: str, churn_risk: float) -> Persona:
    if rng.random() < churn_risk * 0.5:          # risk 0.8 -> 40% of their calls
        return "likely_to_churn"
    weights = MOOD_WEIGHTS[queue]
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def random_customer_ids(rng: random.Random, count: int) -> list[str]:
    with get_engine().connect() as conn:
        ids = list(conn.execute(select(customers.c.customer_id).order_by(customers.c.customer_id)).scalars())
    return rng.sample(ids, count)


def random_call(rng: random.Random, call_id: str, customer_id: str | None = None,
                queue: str | None = None) -> CallState:
    """A new call: a customer (random if not given), a reason fitting their products, a mood."""
    customer_id = customer_id or random_customer_ids(rng, 1)[0]
    facts = customer_facts(customer_id)
    reason = pick_reason(rng, facts, queue)
    persona = pick_persona(rng, REASONS[reason].queue, facts["churn_risk"])
    return CallState(call_id=call_id, customer_id=customer_id, persona=persona, call_reason=reason)
