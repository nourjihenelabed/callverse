"""Advisor: answers the call and asks the complaint specialist for help when needed.

STUB: fixed sentences so the whole call can run without an AI model.
The owner replaces advisor_node with the real worker, keeping the same input and output.

Input: the CallState. Output: a dict with the fields it changes:
- "intent" and "is_complaint" when it understands the request
- "messages": a list with the advisor's new Message
- "tools_used": a list with the ToolCalls it made

To ask the complaint specialist for help: return {"is_complaint": True} with no message.
The graph sends the call to the specialist, then back here with state.complaint_result filled.
"""

from callverse.state import CallState, Message, ToolCall


def advisor_node(state: CallState) -> dict:
    is_complaint = state.call_reason.startswith("complaint")

    # a complaint the specialist has not handled yet: ask for help first
    if is_complaint and state.complaint_result is None:
        return {"intent": state.call_reason, "is_complaint": True}

    # the specialist answered: pass the fix on to the customer
    if state.complaint_result is not None:
        actions = "; ".join(state.complaint_result.actions)
        text = f"Je comprends votre mécontentement. Voici ce que je vous propose : {actions}."
        return {"messages": [Message(role="advisor", content=text)]}

    # an everyday question
    return {
        "intent": state.call_reason,
        "messages": [Message(role="advisor", content="Votre facture comprend votre forfait et vos options.")],
        "tools_used": [ToolCall(name="search_policies", arguments={"question": state.call_reason})],
    }
