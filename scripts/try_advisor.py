"""Talk to the advisor with the real AI model, database and rules search.

Run it from the project folder, with Ollama running, the database filled
(python scripts/seed_data.py) and the rules indexed (python scripts/build_index.py):

    python scripts/try_advisor.py                      # 4 prepared questions
    python scripts/try_advisor.py "Combien coûte l'option TV ?"
    python scripts/try_advisor.py --chat               # type your own messages, one per line
    python scripts/try_advisor.py --customer 7590-VHVEG "Combien je paie par mois ?"

For each message it shows what the advisor understood, the tools it used and its answer.
A complaint goes to the complaint specialist first, as in a real call. Note: the specialist
opens a real ticket in your database for it.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.agents.advisor.agent import advisor_node  # noqa: E402
from callverse.agents.complaint.steps import complaint_node  # noqa: E402
from callverse.state import CallState, Message  # noqa: E402

DEFAULT_CUSTOMER = "7590-VHVEG"     # the first customer of the IBM file
QUESTIONS = [
    "Bonjour, combien coûte la Box Fibre ?",
    "Bonjour, combien je paie par mois et quelles offres j'ai ?",
    "J'ai perdu ma carte SIM, qu'est-ce que je dois faire ?",
    "Ma facture a doublé ce mois-ci, c'est inadmissible, je veux un remboursement !",
]


def merge(state: CallState, update: dict) -> CallState:
    """Apply a worker's update, like the graph does (messages and tools_used are added)."""
    data = state.model_dump()
    for key, value in update.items():
        if key in ("messages", "tools_used"):
            data[key] = data[key] + [v.model_dump() for v in value]
        else:
            data[key] = value.model_dump() if hasattr(value, "model_dump") else value
    return CallState.model_validate(data)


def one_turn(state: CallState) -> CallState:
    start = time.time()
    update = advisor_node(state)
    print(f"  understood: intent={update.get('intent', state.intent)}  "
          f"complaint={update.get('is_complaint', state.is_complaint)}   ({time.time() - start:.0f} s)")
    state = merge(state, update)

    if state.is_complaint and state.complaint_result is None and not update.get("messages"):
        start = time.time()
        state = merge(state, complaint_node(state))
        r = state.complaint_result
        print(f"  COMPLAINT SPECIALIST ({time.time() - start:.0f} s): {r.category}, severity {r.severity}, "
              f"compensation {r.compensation_cents / 100:.2f} EUR, escalate={r.escalate} {r.escalation_reasons}")
        print(f"    actions: {r.actions}")
        print(f"    rules: {r.rules_cited}   ticket: {r.ticket_id}")
        update = advisor_node(state)
        state = merge(state, update)

    for call in update.get("tools_used", []):
        print(f"  tool: {call.name}({call.arguments})")
    if update.get("messages"):
        print(f"  ADVISOR: {update['messages'][0].content}")
    return state


def new_call(customer: str, text: str) -> CallState:
    return CallState(call_id="try", customer_id=customer, persona="calm", call_reason="unknown",
                     messages=[Message(role="client", content=text)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--customer", default=DEFAULT_CUSTOMER)
    parser.add_argument("--chat", action="store_true", help="a whole conversation, typed line by line")
    args = parser.parse_args()

    if args.chat:
        print(f"Customer {args.customer}. Type your messages (empty line to stop).")
        state = None
        while text := input("\nCLIENT: ").strip():
            if state is None:
                state = new_call(args.customer, text)
            else:
                state = state.model_copy(update={"messages": [*state.messages, Message(role="client", content=text)]})
            state = one_turn(state)
        return

    for question in [" ".join(args.question)] if args.question else QUESTIONS:
        print(f"\nCLIENT ({args.customer}): {question}")
        one_turn(new_call(args.customer, question))


if __name__ == "__main__":
    main()
