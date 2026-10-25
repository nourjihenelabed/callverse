"""Talk to the advisor with the real AI model, database and rules search.

Run it from the project folder, with Ollama running, the database filled
(python scripts/seed_data.py) and the rules indexed (python scripts/build_index.py):

    python scripts/try_advisor.py                      # 4 prepared questions
    python scripts/try_advisor.py "Combien coûte l'option TV ?"
    python scripts/try_advisor.py --chat               # type your own messages, one per line
    python scripts/try_advisor.py --customer 7590-VHVEG "Combien je paie par mois ?"

For each message it shows what the advisor understood, the tools it used and its answer.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.agents.advisor.agent import advisor_node  # noqa: E402
from callverse.state import CallState, Message  # noqa: E402

DEFAULT_CUSTOMER = "7590-VHVEG"     # the first customer of the IBM file
QUESTIONS = [
    "Bonjour, combien coûte la Box Fibre ?",
    "Bonjour, combien je paie par mois et quelles offres j'ai ?",
    "J'ai perdu ma carte SIM, qu'est-ce que je dois faire ?",
    "Ma facture a doublé ce mois-ci, c'est inadmissible, je veux un remboursement !",
]


def one_turn(state: CallState) -> CallState:
    start = time.time()
    update = advisor_node(state)
    seconds = time.time() - start

    print(f"  understood: intent={update.get('intent', state.intent)}  "
          f"complaint={update.get('is_complaint', state.is_complaint)}   ({seconds:.0f} s)")
    for call in update.get("tools_used", []):
        print(f"  tool: {call.name}({call.arguments})")
    if update.get("messages"):
        print(f"  ADVISOR: {update['messages'][0].content}")
    elif update.get("is_complaint"):
        print("  -> sent to the complaint specialist (step 4 of the plan, not built yet)")

    data = state.model_dump()
    for key, value in update.items():
        data[key] = data[key] + [v.model_dump() for v in value] if key in ("messages", "tools_used") else value
    return CallState.model_validate(data)


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
