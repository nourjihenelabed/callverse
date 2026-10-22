"""Pretend customer: plays the person who calls, with one of six moods.

STUB: fixed sentences so the whole call can run without an AI model.
The owner replaces client_node with the real worker, keeping the same input and output.

Input: the CallState. Output: a dict with the fields it changes:
- "messages": a list with the customer's new Message
- "turn": state.turn + 1
- at the end of the call, also "outcome" and "satisfaction"
"""

from callverse.state import CallState, Message


def client_node(state: CallState) -> dict:
    # first turn: say why you are calling
    if not state.messages:
        if state.call_reason.startswith("complaint"):
            text = "Bonjour, ma facture est beaucoup trop élevée ce mois-ci, ce n'est pas normal."
        else:
            text = "Bonjour, je voudrais comprendre ma dernière facture."
        return {"messages": [Message(role="client", content=text)], "turn": state.turn + 1}

    # the advisor answered: accept the answer and end the call
    return {
        "messages": [Message(role="client", content="D'accord, merci beaucoup. Au revoir.")],
        "turn": state.turn + 1,
        "outcome": "resolved",
        "satisfaction": 4,
    }
