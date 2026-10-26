"""Churn model: the chance that a customer leaves, learned from the real IBM Telco answers.

It replaces the hand-written placeholder rule of scripts/seed_data.py. The result is the
`churn_risk` column: the complaint specialist sends a customer to a supervisor at 0.70 or more,
and the pretend customer often calls as `likely_to_churn` when it is high. The advisor never sees it.

The model is a logistic regression, written with numpy only (no extra package):
- easy to explain: each input has one weight, and each customer's risk can be split into the
  inputs that raised or lowered it (`reasons`);
- trained on 80% of the customers and tested on the other 20% it never saw (AUC, accuracy, Brier score),
  against the old placeholder rule on the same customers.

Inputs: only real IBM facts (contract, tenure, payment, offers, bill, household). Never the generated
complaints: they were made using the churn label, so they would leak the answer into the model.

Usage (scripts/seed_data.py does it):
    model = train(customer_rows, product_rows)
    risk = model.predict(customer, codes)
"""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_model.json"
SEED = 42
TEST_SHARE = 0.2
STEPS, LEARNING_RATE, L2 = 3000, 0.2, 1e-3

OPTIONS = ("option_securite", "option_sauvegarde", "option_protection_appareil",
           "option_assistance_premium", "option_tv", "option_cinema")

# (name shown in explanations, how to compute it from a customer row and its product codes)
FEATURES = [
    ("months as a customer", lambda c, p: c["tenure_months"]),
    ("months as a customer (log)", lambda c, p: math.log1p(c["tenure_months"])),
    ("monthly bill (€)", lambda c, p: c["monthly_charge_cents"] / 100),
    ("12-month contract", lambda c, p: c["contract"] == "engagement_12_mois"),
    ("24-month contract", lambda c, p: c["contract"] == "engagement_24_mois"),
    ("pays by direct debit", lambda c, p: c["payment_method"] == "prelevement"),
    ("pays by card", lambda c, p: c["payment_method"] == "carte_bancaire"),
    ("pays by cheque", lambda c, p: c["payment_method"] == "cheque"),
    ("paperless bill", lambda c, p: c["paperless_billing"]),
    ("senior", lambda c, p: c["senior_citizen"]),
    ("has a partner", lambda c, p: c["has_partner"]),
    ("has dependents", lambda c, p: c["has_dependents"]),
    ("Box Fibre", lambda c, p: "box_fibre" in p),
    ("Box ADSL", lambda c, p: "box_adsl" in p),
    ("mobile plan", lambda c, p: "forfait_mobile" in p),
    ("extra mobile line", lambda c, p: "ligne_supplementaire" in p),
] + [(code.replace("option_", "option ").replace("_", " "), lambda c, p, code=code: code in p) for code in OPTIONS]
NAMES = [name for name, _ in FEATURES]


NUMERIC = {"months as a customer", "monthly bill (€)"}


def describe(index: int, value: float) -> str:
    """A fact as a person would say it: "no mobile plan", "monthly bill (€): 89.9"."""
    name = NAMES[index]
    if name in NUMERIC:
        return f"{name}: {value:g}"
    return name if value else f"no {name}"


def vector(customer: dict, codes) -> list[float]:
    codes = set(codes)
    return [float(f(customer, codes)) for _, f in FEATURES]


@dataclass
class ChurnModel:
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    metrics: dict = field(default_factory=dict)

    def _z(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=float) - np.asarray(self.mean)) / np.asarray(self.std)

    def predict(self, customer: dict, codes) -> float:
        return float(_sigmoid(self._z(vector(customer, codes)) @ np.asarray(self.weights) + self.bias))

    def reasons(self, customer: dict, codes, top: int = 3) -> list[tuple[str, float]]:
        """The facts that raise this customer's risk the most, with their contribution to the log-odds.
        e.g. [("months as a customer: 1", 1.68), ("no mobile plan", 0.72), ("no 24-month contract", 0.39)]"""
        values = vector(customer, codes)
        parts = list(self._z(values) * np.asarray(self.weights))
        # tenure is two inputs (months and log of months): explain them as one fact
        tenure, log = NAMES.index("months as a customer"), NAMES.index("months as a customer (log)")
        parts[tenure] += parts[log]
        parts[log] = 0.0
        order = [i for i in np.argsort([-p for p in parts]) if parts[i] > 0][:top]
        return [(describe(i, values[i]), round(float(parts[i]), 2)) for i in order]

    def weights_table(self) -> list[tuple[str, float]]:
        """Each input and its weight (per standard deviation), strongest first. + raises the risk."""
        return sorted(zip(NAMES, (round(w, 3) for w in self.weights)), key=lambda item: -abs(item[1]))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"features": NAMES, "mean": self.mean, "std": self.std, "weights": self.weights,
                                    "bias": self.bias, "metrics": self.metrics}, indent=2), encoding="utf-8")


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit(x: np.ndarray, y: np.ndarray) -> ChurnModel:
    """Logistic regression by gradient descent on standardized inputs (same result every time)."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    w, b = np.zeros(z.shape[1]), 0.0
    for _ in range(STEPS):
        error = _sigmoid(z @ w + b) - y
        w -= LEARNING_RATE * (z.T @ error / len(y) + L2 * w)
        b -= LEARNING_RATE * error.mean()
    return ChurnModel(mean.tolist(), std.tolist(), w.tolist(), float(b))


# ---------------------------------------------------------------- evaluation

def auc(scores, labels) -> float:
    """Chance that a random churner gets a higher score than a random customer who stayed."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):                      # average ranks for ties
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return float("nan")
    rank_sum = sum(r for r, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def scores_report(scores, labels, threshold: float = 0.5) -> dict:
    predicted = [s >= threshold for s in scores]
    tp = sum(p and y for p, y in zip(predicted, labels))
    fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels))
    return {
        "auc": round(auc(scores, labels), 3),
        "accuracy": round(sum(p == bool(y) for p, y in zip(predicted, labels)) / len(labels), 3),
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "brier": round(sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels), 3),
    }


def placeholder_rule(customer: dict) -> float:
    """The old hand-written rule (without the generated complaints), kept to compare."""
    risk = 0.10 + 0.30 * (customer["contract"] == "sans_engagement") + 0.20 * (customer["tenure_months"] < 12)
    return risk + 0.10 * (customer["payment_method"] == "paiement_en_ligne")


# ---------------------------------------------------------------- training on the customers

def split(customer_ids: list[str], seed: int = SEED) -> set[str]:
    """The test customers: 20%, the same every time."""
    ids = sorted(customer_ids)
    random.Random(seed).shuffle(ids)
    return set(ids[: int(len(ids) * TEST_SHARE)])


def train(customer_rows: list[dict], product_rows: list[dict]) -> ChurnModel:
    """Learns on 80% of the customers, measures on the other 20%, then refits on everyone."""
    codes = {}
    for p in product_rows:
        codes.setdefault(p["customer_id"], set()).add(p["product_code"])
    x = np.array([vector(c, codes.get(c["customer_id"], ())) for c in customer_rows])
    y = np.array([float(c["churned"]) for c in customer_rows])
    test_ids = split([c["customer_id"] for c in customer_rows])
    is_test = np.array([c["customer_id"] in test_ids for c in customer_rows])

    model = fit(x[~is_test], y[~is_test])
    test = [c for c in customer_rows if c["customer_id"] in test_ids]
    labels = [int(c["churned"]) for c in test]
    scores = [model.predict(c, codes.get(c["customer_id"], ())) for c in test]
    metrics = {
        "train_customers": int((~is_test).sum()), "test_customers": len(test),
        "churn_rate": round(float(y.mean()), 3),
        "model_on_test": scores_report(scores, labels),
        "placeholder_rule_on_test": scores_report([placeholder_rule(c) for c in test], labels),
    }

    final = fit(x, y)                         # the risk written in the database uses every customer
    final.metrics = metrics
    return final


def risk_for_database(model: ChurnModel, customer: dict, codes) -> float:
    return round(min(model.predict(customer, codes), 0.95), 2)
