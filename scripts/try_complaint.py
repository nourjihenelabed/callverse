"""Runs the complaint specialist once on a complaint you type, and shows the whole RAG on screen.

You see, in order:
    Retrieval  - each question sent to the rules search, and the sections it found (with scores)
    Generation - the full prompt sent to the AI model (with the rules pasted in), and its raw answer
    Result     - what the specialist decides once the plain-code checks have run

Uses the real AI model and the real rules search, but NOT the database:
the customer has no earlier complaints and no ticket is written.

Needs Ollama (llama3.1 and nomic-embed-text) and the rules index:  python scripts/build_index.py
Run it from the project folder:
    python scripts/try_complaint.py
    python scripts/try_complaint.py technical "Je n'ai plus internet depuis 5 jours."
The first word is the category: billing, technical, contract, service_quality, termination, delivery.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.agents.complaint import rules, steps  # noqa: E402
from callverse.knowledge import search_policies  # noqa: E402
from callverse.llm import get_llm  # noqa: E402
from callverse.state import CallState, Message  # noqa: E402

DEFAULT = ("technical", "Je n'ai plus internet depuis 5 jours, c'est inadmissible. J'ai la Box Fibre.")

# which step asks the model for which answer
STEP_NAMES = {steps._Classification: "Step 1 - classify", steps._Draft: "Step 4 - draft the fix (generation)"}


def title(text: str) -> None:
    print(f"\n{'=' * 80}\n{text}\n{'=' * 80}")


def show_search(question: str) -> list[dict]:
    """The real rules search, printing the question and what came back."""
    if not show_search.started:
        title("Step 3 - search the rules (retrieval)")
        show_search.started = True
    hits = search_policies(question)
    print(f"\nQuestion: {question}")
    for hit in hits:
        print(f"  {hit['score']:.3f}  {hit['source']} > {hit['section']}")
    return hits


show_search.started = False


class ShowModel:
    """The real model, printing every prompt it gets and every answer it gives."""

    def __init__(self, llm):
        self.llm = llm

    def with_structured_output(self, schema):
        runner = self.llm.with_structured_output(schema)
        name = STEP_NAMES.get(schema, schema.__name__)

        class Show:
            def invoke(self, prompt):
                title(f"{name}: PROMPT sent to the model")
                print(prompt)
                answer = runner.invoke(prompt)
                title(f"{name}: raw ANSWER of the model")
                print(answer.model_dump_json(indent=2))
                return answer

        return Show()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    category, text = (sys.argv[1], " ".join(sys.argv[2:])) if len(sys.argv) > 2 else DEFAULT
    if category not in rules.REQUIRED_ACTIONS:
        sys.exit(f"Unknown category {category!r}. Use one of: {', '.join(rules.REQUIRED_ACTIONS)}")

    tools = steps.Tools(
        llm=ShowModel(get_llm(temperature=0)),
        get_history=lambda customer_id, category: steps.History(),     # no database
        search=show_search,
        open_ticket=lambda state, result, history: "(no ticket: database not used)",
        update_ticket=lambda ticket_id, new_state: None,
    )
    state = CallState(call_id="try", customer_id="try", persona="dissatisfied",
                      call_reason=f"complaint_{category}", messages=[Message(role="client", content=text)])

    print(f"Complaint ({category}): {text}")
    result = steps.handle_complaint(state, tools)

    title("Result after the plain-code checks (steps 5-7)")
    print(f"Category:     {result.category}   severity {result.severity}")
    print(f"Compensation: {result.compensation_cents / 100:.2f} EUR ({result.compensation_cents} cents)")
    print("Actions:")
    for action in result.actions:
        print(f"  - {action}")
    print("Rules cited:")
    for rule in result.rules_cited:
        print(f"  - {rule}")
    print(f"Supervisor:   {'yes: ' + ', '.join(result.escalation_reasons) if result.escalate else 'no'}")


if __name__ == "__main__":
    main()
