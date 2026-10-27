"""Supervisor (retention): decides what to do when a customer wants to leave.

The rules come from the company documents, so this is plain code (no AI model):
- agent_guidelines.md, "Client qui veut partir": explain the cancellation conditions, and hand the
  case to a supervisor if the churn risk is high.
- compensation_grid.md, "Offres de fidélisation": only a supervisor may offer, for a customer at high
  risk, up to 3 free months on an option (e.g. TV or Cinéma & Séries), or a free move from the
  Box ADSL to the Box Fibre if the home is eligible.

High risk = churn_risk at or above the same threshold as the complaint rules (0.70), from the churn model.

Input: the CallState. Output: {"retention": RetentionResult(...)}.
The customer's account comes from get_facts(), so tests can replace it.
"""

from sqlalchemy import select

from callverse.agents.complaint.rules import CHURN_RISK_LIMIT
from callverse.state import CallState, RetentionResult

RULE_OFFERS = "compensation_grid.md > Offres de fidélisation"
RULE_LEAVING = "agent_guidelines.md > Adapter son ton au client (client qui veut partir)"

# the options a loyalty offer can make free for 3 months, in order of preference
FREE_OPTIONS = (("option_tv", "TV"), ("option_cinema", "Cinéma & Séries"))


def get_facts(customer_id: str) -> dict:
    """The churn risk and the product codes of the customer. Tests replace this function."""
    from callverse.db import customer_products, customers, get_engine

    with get_engine().connect() as conn:
        risk = conn.execute(select(customers.c.churn_risk).where(customers.c.customer_id == customer_id)).scalar()
        codes = set(conn.execute(select(customer_products.c.product_code)
                                 .where(customer_products.c.customer_id == customer_id)).scalars())
    return {"churn_risk": risk or 0.0, "codes": codes}


def decide(churn_risk: float, codes) -> RetentionResult:
    codes = set(codes)
    if churn_risk < CHURN_RISK_LIMIT:
        # not high risk: the advisor explains the cancellation conditions, no offer
        return RetentionResult(high_risk=False, supervisor=False, rules_cited=[RULE_LEAVING])

    if "box_adsl" in codes:
        return RetentionResult(
            high_risk=True, supervisor=True, offer_key="Box Fibre",
            offer="le passage gratuit de votre Box ADSL à la Box Fibre, au même prix, si votre logement est éligible",
            rules_cited=[RULE_LEAVING, RULE_OFFERS])
    if any(code.startswith("box_") for code in codes):
        name = next((label for code, label in FREE_OPTIONS if code not in codes), FREE_OPTIONS[-1][1])
        return RetentionResult(
            high_risk=True, supervisor=True, offer_key="3 mois offerts",
            offer=f"3 mois offerts sur l'option {name}",
            rules_cited=[RULE_LEAVING, RULE_OFFERS])
    # mobile only: the grid has no offer for it; the supervisor calls the customer back
    return RetentionResult(high_risk=True, supervisor=True, rules_cited=[RULE_LEAVING])


def retention_node(state: CallState) -> dict:
    try:
        facts = get_facts(state.customer_id)
    except Exception:                   # no database: treat as not high risk (explain conditions only)
        facts = {"churn_risk": 0.0, "codes": set()}
    return {"retention": decide(facts["churn_risk"], facts["codes"])}
