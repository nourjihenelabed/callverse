"""Explains the churn model (callverse/churn.py): how good it is, what drives it, and why a customer is at risk.

Run it from the project folder (needs the IBM CSV in data/raw/):
    python scripts/explain_churn.py
    python scripts/explain_churn.py 7590-VHVEG 9237-HQITU      # explain these customers

The model is trained exactly as scripts/seed_data.py trains it, so the numbers are the same.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from callverse import churn  # noqa: E402

spec = importlib.util.spec_from_file_location("seed_data", ROOT / "scripts" / "seed_data.py")
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)


def main() -> None:
    rows, products = seed.build_customers(seed.load_ibm(seed.find_ibm_csv()))
    model = churn.train(rows, products)
    codes = {}
    for p in products:
        codes.setdefault(p["customer_id"], set()).add(p["product_code"])

    m = model.metrics
    print(f"Trained on {m['train_customers']} customers, tested on {m['test_customers']} it never saw "
          f"({m['churn_rate']:.1%} of customers left).\n")
    print(f"{'':<22}{'AUC':>7}{'accuracy':>10}{'precision':>11}{'recall':>8}{'Brier':>7}")
    for name, key in (("Churn model", "model_on_test"), ("Old placeholder rule", "placeholder_rule_on_test")):
        r = m[key]
        print(f"{name:<22}{r['auc']:>7}{r['accuracy']:>10.1%}{r['precision']:>11.1%}{r['recall']:>8.1%}{r['brier']:>7}")

    print("\nWhat drives the risk (weight per standard deviation; + raises it, - lowers it):")
    for name, weight in model.weights_table():
        print(f"  {weight:+.2f}  {'#' * round(abs(weight) * 20):<20} {name}")

    risks = sorted(((churn.risk_for_database(model, c, codes.get(c["customer_id"], ())), c) for c in rows),
                   key=lambda item: -item[0])
    print(f"\n{sum(r >= 0.70 for r, _ in risks)} customers at 0.70 or more (sent to a supervisor when they complain).")
    wanted = sys.argv[1:] or [c["customer_id"] for _, c in risks[:3]]
    by_id = {c["customer_id"]: c for c in rows}
    for customer_id in wanted:
        c = by_id[customer_id]
        reasons = ", ".join(f"{name} (+{value})" for name, value in model.reasons(c, codes.get(customer_id, ())))
        print(f"  {customer_id}: risk {churn.risk_for_database(model, c, codes.get(customer_id, ())):.2f}, "
              f"left in reality: {'yes' if c['churned'] else 'no'}. Why: {reasons}")


if __name__ == "__main__":
    main()
