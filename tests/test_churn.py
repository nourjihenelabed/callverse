"""Checks the churn model (callverse/churn.py) on made-up customers whose answer we control.

The real check on the 7,043 IBM customers is in tests/test_seed.py (skipped without the CSV).
"""

import random

import pytest

from callverse import churn


def customer(i: int, rng: random.Random) -> tuple[dict, list[str], bool]:
    """Short tenure, no commitment and fibre make leaving likely; the answer is drawn from that."""
    tenure = rng.randint(0, 72)
    contract = rng.choice(["sans_engagement", "engagement_12_mois", "engagement_24_mois"])
    codes = ["box_fibre" if rng.random() < 0.5 else "box_adsl"] + (["forfait_mobile"] if rng.random() < 0.7 else [])
    logit = 1.0 - 0.06 * tenure + 1.5 * (contract == "sans_engagement") + 0.8 * ("box_fibre" in codes)
    churned = rng.random() < 1 / (1 + 2.718 ** -logit)
    row = {"customer_id": f"C{i:05d}", "tenure_months": tenure, "contract": contract,
           "payment_method": rng.choice(["prelevement", "carte_bancaire", "cheque", "paiement_en_ligne"]),
           "paperless_billing": rng.random() < 0.5, "senior_citizen": rng.random() < 0.2,
           "has_partner": rng.random() < 0.5, "has_dependents": rng.random() < 0.3,
           "monthly_charge_cents": 2499 + 1000 * ("box_fibre" in codes) + 1499 * ("forfait_mobile" in codes),
           "churned": churned}
    return row, codes, churned


@pytest.fixture(scope="module")
def data():
    rng = random.Random(7)
    made = [customer(i, rng) for i in range(2000)]
    rows = [m[0] for m in made]
    products = [{"customer_id": r["customer_id"], "product_code": c} for r, codes, _ in made for c in codes]
    return rows, products, {r["customer_id"]: codes for r, codes, _ in made}


def test_model_learns_the_real_pattern_and_beats_a_coin(data):
    rows, products, _ = data
    model = churn.train(rows, products)
    test = model.metrics["model_on_test"]
    assert test["auc"] > 0.75 and model.metrics["test_customers"] == 400
    weights = dict(model.weights_table())
    assert weights["24-month contract"] < 0 and weights["Box Fibre"] > 0       # the directions we built in


def test_training_gives_the_same_model_every_time(data):
    rows, products, _ = data
    assert churn.train(rows, products).weights == churn.train(rows, products).weights


def test_test_customers_are_never_used_to_measure_on_training_data():
    ids = [f"C{i}" for i in range(100)]
    test = churn.split(ids)
    assert len(test) == 20 and test == churn.split(list(reversed(ids)))       # same split whatever the order


def test_reasons_explain_a_risky_customer(data):
    rows, products, codes = data
    model = churn.train(rows, products)
    risky = {"customer_id": "X", "tenure_months": 1, "contract": "sans_engagement", "payment_method": "cheque",
             "paperless_billing": False, "senior_citizen": False, "has_partner": False, "has_dependents": False,
             "monthly_charge_cents": 3499}
    assert model.predict(risky, ["box_fibre"]) > 0.5
    names = [name for name, _ in model.reasons(risky, ["box_fibre"])]
    assert "months as a customer: 1" in names or "no 24-month contract" in names
    assert all(value > 0 for _, value in model.reasons(risky, ["box_fibre"]))


def test_auc_on_known_cases():
    assert churn.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert churn.auc([0.9, 0.8], [1, 1]) != churn.auc([0.9, 0.8], [1, 1])      # one class only: not defined


def test_risk_in_the_database_is_capped(data):
    rows, products, codes = data
    model = churn.train(rows, products)
    assert all(0 <= churn.risk_for_database(model, r, codes[r["customer_id"]]) <= 0.95 for r in rows[:200])


def test_saved_model_has_its_test_results(data, tmp_path):
    rows, products, _ = data
    model = churn.train(rows, products)
    model.save(tmp_path / "churn.json")
    text = (tmp_path / "churn.json").read_text(encoding="utf-8")
    assert '"model_on_test"' in text and '"placeholder_rule_on_test"' in text
