"""Fills the PostgreSQL database with the same data on every computer.

Tables: customers (from IBM Telco), customer_products, complaints, tickets.
Uses a fixed random seed so everyone gets identical pretend data.

Run it from the project folder:
    python scripts/seed_data.py

It deletes the tables and builds them again, so you can run it as often as you like.
It needs the IBM Telco CSV in data/raw/ (see data/raw/README.md) and your .env file.
"""

import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42                                   # change it only with team agreement: everyone's data changes
REFERENCE_DATE = datetime(2026, 1, 1)       # "today" for the generated history, fixed so data is identical

# Monthly prices in euro cents. Must match data/policies/catalogue.md (a test checks it).
CATALOGUE = {
    "box_fibre": 3499,
    "box_adsl": 2499,
    "forfait_mobile": 1499,
    "ligne_supplementaire": 999,
    "option_securite": 399,
    "option_sauvegarde": 299,
    "option_protection_appareil": 499,
    "option_assistance_premium": 599,
    "option_tv": 799,
    "option_cinema": 999,
}

# IBM column (value "Yes") -> CallVerse option
IBM_OPTIONS = {
    "OnlineSecurity": "option_securite",
    "OnlineBackup": "option_sauvegarde",
    "DeviceProtection": "option_protection_appareil",
    "TechSupport": "option_assistance_premium",
    "StreamingTV": "option_tv",
    "StreamingMovies": "option_cinema",
}

CONTRACTS = {
    "Month-to-month": "sans_engagement",
    "One year": "engagement_12_mois",
    "Two year": "engagement_24_mois",
}

PAYMENT_METHODS = {
    "Bank transfer (automatic)": "prelevement",
    "Credit card (automatic)": "carte_bancaire",
    "Electronic check": "paiement_en_ligne",
    "Mailed check": "cheque",
}

# ---- complaint generation. PLACEHOLDER shares: to calibrate on FCC and Médiateur data.
COMPLAINT_CHANCE = {"churned": 0.45, "stayed": 0.20}   # chance a customer has past complaints
COMPLAINT_COUNT_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}
CATEGORY_WEIGHTS = {
    "billing": 0.30,
    "technical": 0.30,
    "contract": 0.15,
    "service_quality": 0.12,
    "termination": 0.08,
    "delivery": 0.05,
}
SEVERITY_WEIGHTS = {1: 0.35, 2: 0.35, 3: 0.20, 4: 0.10}
UNRESOLVED_SHARE = 0.25                     # about a quarter of complaints stay open
COMPENSATION_BY_SEVERITY = {1: 0, 2: 500, 3: 1000, 4: 2000}   # cents, when a resolved complaint got one
REPEAT_SHARE = 0.03                         # customers given 2 recent unresolved complaints of the same kind,
                                            # so the "3rd complaint -> supervisor" rule gets tested

DESCRIPTIONS = {
    "billing": [
        "Montant de la facture plus élevé que prévu.",
        "Option facturée alors qu'elle n'a pas été demandée.",
        "Prélèvement effectué deux fois le même mois.",
    ],
    "technical": [
        "Coupures internet fréquentes le soir.",
        "Débit beaucoup plus lent que promis.",
        "Box qui redémarre seule plusieurs fois par jour.",
    ],
    "contract": [
        "Changement d'offre non appliqué.",
        "Conditions d'engagement mal expliquées à la souscription.",
    ],
    "service_quality": [
        "Attente très longue au service client.",
        "Conseiller qui n'a pas rappelé comme promis.",
    ],
    "termination": [
        "Frais de résiliation contestés.",
        "Résiliation demandée mais toujours facturée.",
    ],
    "delivery": [
        "Box jamais reçue après la souscription.",
        "Technicien absent au rendez-vous d'installation.",
    ],
}

UNRESOLVED_TICKET_STATES = ["open", "solution_proposed", "refused", "reopened"]


# ---------------------------------------------------------------- reading the CSV

def find_ibm_csv() -> Path:
    raw = PROJECT_ROOT / "data" / "raw"
    found = sorted(raw.glob("*Telco-Customer-Churn*.csv"))
    if not found:
        raise FileNotFoundError(
            f"No IBM Telco CSV in {raw}. Download it (see data/raw/README.md) and put it there."
        )
    return found[0]


def load_ibm(path: Path) -> list[dict]:
    """Reads the IBM CSV as a list of rows (one dict per customer), sorted by customer id."""
    with open(path, newline="", encoding="utf-8") as f:
        return sorted(csv.DictReader(f), key=lambda row: row["customerID"])


# ---------------------------------------------------------------- customers

def products_for(row) -> list[str]:
    """Turns one IBM row into CallVerse product codes."""
    products = []
    if row["InternetService"] == "Fiber optic":
        products.append("box_fibre")
    elif row["InternetService"] == "DSL":
        products.append("box_adsl")
    if row["PhoneService"] == "Yes":
        products.append("forfait_mobile")
    if row["MultipleLines"] == "Yes":
        products.append("ligne_supplementaire")
    for column, code in IBM_OPTIONS.items():
        if row[column] == "Yes":
            products.append(code)
    return products


def build_customers(ibm_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    customer_rows, product_rows = [], []
    for row in ibm_rows:
        products = products_for(row)
        monthly = sum(CATALOGUE[p] for p in products)
        tenure = int(row["tenure"])
        customer_rows.append({
            "customer_id": row["customerID"],
            "gender": "female" if row["gender"] == "Female" else "male",
            "senior_citizen": str(row["SeniorCitizen"]) == "1",
            "has_partner": row["Partner"] == "Yes",
            "has_dependents": row["Dependents"] == "Yes",
            "tenure_months": tenure,
            "contract": CONTRACTS[row["Contract"]],
            "paperless_billing": row["PaperlessBilling"] == "Yes",
            "payment_method": PAYMENT_METHODS[row["PaymentMethod"]],
            "monthly_charge_cents": monthly,
            "total_charge_cents": monthly * tenure,
            "churn_risk": None,               # filled after complaints are generated
            "churned": row["Churn"] == "Yes",
        })
        product_rows += [{"customer_id": row["customerID"], "product_code": p} for p in products]
    return customer_rows, product_rows


def churn_risk(customer: dict, unresolved_complaints: int) -> float:
    """PLACEHOLDER rule until the team trains a churn model. Never uses the churned label."""
    risk = 0.10
    if customer["contract"] == "sans_engagement":
        risk += 0.30
    if customer["tenure_months"] < 12:
        risk += 0.20
    if customer["payment_method"] == "paiement_en_ligne":
        risk += 0.10
    risk += 0.10 * unresolved_complaints
    return round(min(risk, 0.95), 2)


# ---------------------------------------------------------------- complaints and tickets

def _pick(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _complaint(rng, number, customer_id, category, opened_at, unresolved):
    severity = _pick(rng, SEVERITY_WEIGHTS)
    if unresolved:
        closed_at, compensation = None, 0
        state = rng.choice(UNRESOLVED_TICKET_STATES)
        updated_at = min(opened_at + timedelta(days=rng.randint(1, 10)), REFERENCE_DATE)
    else:
        closed_at = min(opened_at + timedelta(days=rng.randint(1, 20)), REFERENCE_DATE)
        compensation = COMPENSATION_BY_SEVERITY[severity] if rng.random() < 0.5 else 0
        state, updated_at = "resolved", closed_at
    complaint = {
        "complaint_id": f"CMP-{number:06d}",
        "customer_id": customer_id,
        "category": category,
        "severity": severity,
        "description": rng.choice(DESCRIPTIONS[category]),
        "status": "unresolved" if unresolved else "resolved",
        "compensation_cents": compensation,
        "opened_at": opened_at,
        "closed_at": closed_at,
    }
    ticket = {
        "ticket_id": f"TCK-{number:06d}",
        "complaint_id": complaint["complaint_id"],
        "customer_id": customer_id,
        "state": state,
        "created_at": opened_at,
        "updated_at": updated_at,
    }
    return complaint, ticket


def build_complaints(customer_rows: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Generates the same complaints and tickets every time for the same seed."""
    rng = random.Random(seed)
    complaint_rows, ticket_rows = [], []

    def add(customer_id, category, opened_at, unresolved):
        complaint, ticket = _complaint(rng, len(complaint_rows) + 1, customer_id, category, opened_at, unresolved)
        complaint_rows.append(complaint)
        ticket_rows.append(ticket)

    # 1. ordinary history
    for customer in customer_rows:
        if customer["tenure_months"] == 0:
            continue
        chance = COMPLAINT_CHANCE["churned" if customer["churned"] else "stayed"]
        if rng.random() >= chance:
            continue
        max_days = min(customer["tenure_months"] * 30, 730)
        for _ in range(_pick(rng, COMPLAINT_COUNT_WEIGHTS)):
            opened_at = REFERENCE_DATE - timedelta(days=rng.randint(1, max_days))
            add(customer["customer_id"], _pick(rng, CATEGORY_WEIGHTS), opened_at, rng.random() < UNRESOLVED_SHARE)

    # 2. repeat cases: 2 recent unresolved complaints of the same category
    candidates = [c["customer_id"] for c in customer_rows if c["tenure_months"] >= 6]
    for customer_id in rng.sample(candidates, int(len(candidates) * REPEAT_SHARE)):
        category = rng.choice(["billing", "technical"])
        for _ in range(2):
            add(customer_id, category, REFERENCE_DATE - timedelta(days=rng.randint(5, 90)), True)

    # 3. churn risk now that unresolved complaints are known
    unresolved = {}
    for c in complaint_rows:
        if c["status"] == "unresolved":
            unresolved[c["customer_id"]] = unresolved.get(c["customer_id"], 0) + 1
    for customer in customer_rows:
        customer["churn_risk"] = churn_risk(customer, unresolved.get(customer["customer_id"], 0))

    return complaint_rows, ticket_rows


# ---------------------------------------------------------------- writing to the database

def write_all(engine, customer_rows, product_rows, complaint_rows, ticket_rows) -> None:
    from callverse.db import complaints, customer_products, customers, metadata, tickets

    metadata.drop_all(engine)
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(customers.insert(), customer_rows)
        conn.execute(customer_products.insert(), product_rows)
        conn.execute(complaints.insert(), complaint_rows)
        conn.execute(tickets.insert(), ticket_rows)


def main() -> None:
    from callverse.db import get_engine

    path = find_ibm_csv()
    customer_rows, product_rows = build_customers(load_ibm(path))
    complaint_rows, ticket_rows = build_complaints(customer_rows)
    write_all(get_engine(), customer_rows, product_rows, complaint_rows, ticket_rows)

    unresolved = sum(c["status"] == "unresolved" for c in complaint_rows)
    print(f"Read {path.name}")
    print(f"customers:         {len(customer_rows)}")
    print(f"customer_products: {len(product_rows)}")
    print(f"complaints:        {len(complaint_rows)} ({unresolved} unresolved)")
    print(f"tickets:           {len(ticket_rows)}")


if __name__ == "__main__":
    main()
