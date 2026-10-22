"""The database tables, and the one place where workers connect to PostgreSQL.

Shared by everyone. Ask the team before changing it.

Usage in a worker:
    from sqlalchemy import select
    from callverse.db import get_engine, customers

    with get_engine().connect() as conn:
        row = conn.execute(select(customers).where(customers.c.customer_id == "7590-VHVEG")).first()

Money is always in euro cents (2999 = 29.99 EUR).
The tables are created and filled by scripts/seed_data.py.
"""

import os
from functools import lru_cache

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)

from callverse.config import load_settings

metadata = MetaData()

# One row per customer. Built from the IBM Telco dataset, re-priced from data/policies/catalogue.md.
customers = Table(
    "customers",
    metadata,
    Column("customer_id", String(20), primary_key=True),      # IBM id, e.g. "7590-VHVEG"
    Column("gender", String(10)),
    Column("senior_citizen", Boolean),
    Column("has_partner", Boolean),
    Column("has_dependents", Boolean),
    Column("tenure_months", Integer),                           # months as a customer
    Column("contract", String(30)),                             # sans_engagement, engagement_12_mois, engagement_24_mois
    Column("paperless_billing", Boolean),
    Column("payment_method", String(30)),                       # prelevement, carte_bancaire, paiement_en_ligne, cheque
    Column("monthly_charge_cents", Integer),                    # sum of the customer's products in the catalogue
    Column("total_charge_cents", Integer),                      # monthly charge x tenure
    Column("churn_risk", Float),                                # 0 to 1. Simple rule for now, a churn model later.
    Column("churned", Boolean),                                 # IBM label. For evaluation only: agents must not read it.
)

# The products each customer has (codes from the catalogue).
customer_products = Table(
    "customer_products",
    metadata,
    Column("customer_id", String(20), ForeignKey("customers.customer_id"), primary_key=True),
    Column("product_code", String(40), primary_key=True),
)

# Past complaints (generated).
complaints = Table(
    "complaints",
    metadata,
    Column("complaint_id", String(20), primary_key=True),      # e.g. "CMP-000001"
    Column("customer_id", String(20), ForeignKey("customers.customer_id"), index=True),
    Column("category", String(30)),                             # same words as ComplaintCategory in state.py
    Column("severity", Integer),                                # 1 to 4
    Column("description", Text),
    Column("status", String(20)),                               # resolved, unresolved
    Column("compensation_cents", Integer),
    Column("opened_at", DateTime),
    Column("closed_at", DateTime, nullable=True),
)

# One ticket follows each complaint.
tickets = Table(
    "tickets",
    metadata,
    Column("ticket_id", String(20), primary_key=True),         # e.g. "TCK-000001"
    Column("complaint_id", String(20), ForeignKey("complaints.complaint_id"), unique=True),
    Column("customer_id", String(20), ForeignKey("customers.customer_id"), index=True),
    Column("state", String(30)),                                # open, solution_proposed, accepted, refused, resolved, reopened
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
)


@lru_cache
def get_engine():
    """The connection to PostgreSQL. The address comes from the .env file."""
    variable = load_settings()["database"]["url_from_env"]
    url = os.getenv(variable)
    if not url:
        raise RuntimeError(
            f"{variable} is not set. Copy .env.example to .env and put your PostgreSQL password in it."
        )
    return create_engine(url)
