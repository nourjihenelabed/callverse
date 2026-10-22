"""Checks the data recipe in scripts/seed_data.py.

No PostgreSQL needed: the database test uses a temporary SQLite file.
Tests that need the real IBM CSV are skipped if data/raw/ does not have it.
"""

import importlib.util
import re
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("seed_data", ROOT / "scripts" / "seed_data.py")
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)

from callverse.db import complaints, customers, tickets  # noqa: E402
from callverse.state import ComplaintResult  # noqa: E402

COLUMNS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
           "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
           "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
           "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]

FAKE_ROWS = [
    ["0001-AAAAA", "Female", 0, "Yes", "No", 10, "Yes", "Yes", "Fiber optic",
     "Yes", "No", "No", "No", "Yes", "No", "Month-to-month", "Yes", "Electronic check", 90.0, 900.0, "Yes"],
    ["0002-BBBBB", "Male", 1, "No", "No", 0, "No", "No phone service", "DSL",
     "No", "No", "No", "No", "No", "No", "Two year", "No", "Mailed check", 25.0, " ", "No"],
    ["0003-CCCCC", "Male", 0, "No", "Yes", 40, "Yes", "No", "No",
     "No internet service", "No internet service", "No internet service", "No internet service",
     "No internet service", "No internet service", "One year", "No", "Bank transfer (automatic)", 20.0, 800.0, "No"],
]


def fake_rows():
    """The same shape as a CSV row read by load_ibm: every value is text."""
    return [dict(zip(COLUMNS, map(str, values))) for values in FAKE_ROWS]


def test_catalogue_prices_match_catalogue_md():
    text = (ROOT / "data" / "policies" / "catalogue.md").read_text(encoding="utf-8")
    for code, cents in seed.CATALOGUE.items():
        match = re.search(rf"\| {code} \|[^|]*\| ([\d,]+) € \|", text)
        assert match, f"{code} is missing from catalogue.md"
        assert round(float(match.group(1).replace(",", ".")) * 100) == cents, f"price of {code} differs"


def test_customers_are_repriced_from_the_catalogue():
    rows, products = seed.build_customers(fake_rows())
    first = rows[0]
    expected = ["box_fibre", "forfait_mobile", "ligne_supplementaire", "option_securite", "option_tv"]
    assert [p["product_code"] for p in products if p["customer_id"] == "0001-AAAAA"] == expected
    assert first["monthly_charge_cents"] == 3499 + 1499 + 999 + 399 + 799
    assert first["total_charge_cents"] == first["monthly_charge_cents"] * 10
    assert first["contract"] == "sans_engagement" and first["payment_method"] == "paiement_en_ligne"
    assert rows[1]["total_charge_cents"] == 0          # new customer, tenure 0
    assert rows[2]["monthly_charge_cents"] == 1499     # mobile only


def test_same_seed_gives_identical_complaints():
    rows_a, _ = seed.build_customers(fake_rows())
    rows_b, _ = seed.build_customers(fake_rows())
    assert seed.build_complaints(rows_a, seed=7) == seed.build_complaints(rows_b, seed=7)


def test_generated_complaints_fit_the_complaint_contract():
    rows, _ = seed.build_customers(fake_rows())
    complaint_rows, ticket_rows = seed.build_complaints(rows, seed=3)
    for c in complaint_rows:
        ComplaintResult(category=c["category"], severity=c["severity"], actions=[],
                        compensation_cents=c["compensation_cents"])   # same words and limits as state.py
    assert len(ticket_rows) == len(complaint_rows)


def test_writing_to_a_database(tmp_path):
    rows, products = seed.build_customers(fake_rows())
    complaint_rows, ticket_rows = seed.build_complaints(rows, seed=1)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    seed.write_all(engine, rows, products, complaint_rows, ticket_rows)
    seed.write_all(engine, rows, products, complaint_rows, ticket_rows)   # running twice must work
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(customers)).scalar() == 3
        assert conn.execute(select(func.count()).select_from(complaints)).scalar() == len(complaint_rows)
        assert conn.execute(select(func.count()).select_from(tickets)).scalar() == len(ticket_rows)


@pytest.mark.skipif(not list((ROOT / "data" / "raw").glob("*Telco-Customer-Churn*.csv")),
                    reason="IBM CSV not in data/raw/")
def test_full_ibm_file():
    rows, _ = seed.build_customers(seed.load_ibm(seed.find_ibm_csv()))
    complaint_rows, _ = seed.build_complaints(rows)
    assert len(rows) == 7043
    assert all(0 <= r["churn_risk"] <= 0.95 for r in rows)
    repeat = {}
    for c in complaint_rows:
        if c["status"] == "unresolved":
            key = (c["customer_id"], c["category"])
            repeat[key] = repeat.get(key, 0) + 1
    assert sum(n >= 2 for n in repeat.values()) >= 100   # enough cases for the "3rd complaint" rule
