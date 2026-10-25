"""Checks that the complaint specialist finds the right company rules (step 3).

For each test complaint we know which rule section the specialist must read.
The script runs step 3 exactly as the agent does and says if that section came back.

Needs Ollama (nomic-embed-text) and the rules index:  python scripts/build_index.py
Run:  python scripts/eval_complaint_rag.py
"""

import sys

from callverse.agents.complaint import steps
from callverse.knowledge import search_policies
from callverse.state import CallState, Message

# (category, severity, what the customer says, rule sections that answer it: any one is enough)
CASES = [
    ("billing", 2, "On m'a facturé l'option TV ce mois-ci alors que je ne l'ai jamais demandée.",
     ["compensation_grid.md > Erreur de facturation (catégorie facturation)"]),
    ("billing", 3, "J'ai été prélevé deux fois ce mois-ci, ma banque m'a même compté des frais.",
     ["compensation_grid.md > Erreur de facturation (catégorie facturation)"]),
    ("technical", 3, "Je n'ai plus internet depuis 5 jours, c'est inadmissible.",
     ["compensation_grid.md > Coupure de service (catégorie technique)"]),
    ("technical", 2, "Mon internet est très lent depuis une semaine, je ne peux rien faire.",
     ["procedures.md > Débit internet lent", "compensation_grid.md > Coupure de service (catégorie technique)"]),
    ("contract", 2, "J'ai demandé à passer à la Box Fibre il y a un mois et ce n'est toujours pas fait.",
     ["contracts.md > Changement d'offre"]),
    ("contract", 2, "On me dit que je suis engagé 24 mois alors que j'ai signé pour 12 mois.",
     ["contracts.md > Durées d'engagement"]),
    ("termination", 2, "On me facture 49 euros de frais de résiliation pour ma box, je conteste.",
     ["contracts.md > Frais de résiliation de la box"]),
    ("termination", 2, "J'ai résilié mon abonnement et vous continuez à me facturer.",
     ["complaint_procedures.md > Étapes obligatoires par catégorie", "contracts.md > Facturation et paiement"]),
    ("service_quality", 2, "Le conseiller m'a parlé très mal et m'a raccroché au nez.",
     ["compensation_grid.md > Qualité de service"]),
    ("service_quality", 2, "On m'avait promis de me rappeler et personne ne l'a jamais fait.",
     ["compensation_grid.md > Qualité de service"]),
    ("delivery", 2, "Le technicien n'est jamais venu au rendez-vous et personne ne m'a prévenu.",
     ["compensation_grid.md > Livraison et installation"]),
    ("delivery", 2, "Ma box n'est toujours pas arrivée, j'attends depuis trois semaines.",
     ["compensation_grid.md > Livraison et installation"]),
]


def evaluate(search=search_policies) -> list[dict]:
    tools = steps.Tools(llm=None, get_history=None, search=search, open_ticket=None, update_ticket=None)
    results = []
    for category, severity, text, expected in CASES:
        state = CallState(call_id="eval", customer_id="eval", persona="dissatisfied",
                          call_reason=f"complaint_{category}", messages=[Message(role="client", content=text)])
        case = steps.Case(category=category, severity=severity)
        steps.search_rules(state, case, tools)
        found = [steps._label(hit) for hit in case.hits]
        rank = next((n for n, label in enumerate(found, 1) if label in expected), None)
        results.append({"category": category, "text": text, "rank": rank, "found": found})
    return results


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    results = evaluate()
    for r in results:
        mark = f"ok (rank {r['rank']})" if r["rank"] else "MISSED"
        print(f"{mark:14} {r['category']:16} {r['text']}")
        if not r["rank"]:
            print("               found instead:", "; ".join(r["found"]))
    hits = sum(r["rank"] is not None for r in results)
    print(f"\nright rule found for {hits}/{len(results)} complaints")


if __name__ == "__main__":
    main()
