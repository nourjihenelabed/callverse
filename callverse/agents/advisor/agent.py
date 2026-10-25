"""Advisor: answers the call and asks the complaint specialist for help when needed.

How one advisor turn works:
1. Understand: while no complaint has been handled, the AI model reads the customer's
   last message and decides the intent and whether it is a complaint.
2. A complaint the specialist has not seen yet: return {"is_complaint": True} with no
   message. The graph sends the call to the specialist, then back here.
3. Answer: the AI model answers with its tools (search_policies, get_customer,
   get_complaint_history). It may call tools several times, up to MAX_TOOL_ROUNDS,
   then it must answer. When the specialist has decided, its decision is given to the
   model so it explains exactly that to the customer.

Input: the CallState. Output: a dict with the fields it changes:
- "intent" and "is_complaint" when it understands the request
- "messages": a list with the advisor's new Message
- "tools_used": a list with the ToolCalls it made

The AI model comes from get_model(), so tests can replace it with a fake one.
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from callverse.agents.advisor import prompts
from callverse.agents.advisor.tools import build_tools, euros
from callverse.llm import get_llm
from callverse.state import CallState, Message, ToolCall

MAX_TOOL_ROUNDS = 4
FALLBACK_ANSWER = ("Je suis désolé, je n'ai pas pu trouver cette information. "
                   "Je vous propose de vous rappeler dès que j'ai la réponse. Puis-je vous aider pour autre chose ?")


class Understanding(BaseModel):
    intent: str = Field(description="what the customer wants, a few English words with underscores")
    is_complaint: bool = Field(description="true if the customer complains and expects a fix or a gesture")


def get_model():
    """The AI model. Tests replace this function with a fake model."""
    return get_llm()


def advisor_node(state: CallState) -> dict:
    model = get_model()
    update = {}

    # 1. understand the customer's last message (until a complaint has been handled)
    if state.complaint_result is None:
        understood = understand(model, state)
        if state.intent is None or (understood.is_complaint and not state.is_complaint):
            update["intent"] = understood.intent
        update["is_complaint"] = state.is_complaint or understood.is_complaint

        # 2. a complaint: the specialist decides first
        if update["is_complaint"]:
            return update

    # 3. answer the customer, with the tools
    text, calls = answer(model, state)
    update["messages"] = [Message(role="advisor", content=text)]
    if calls:
        update["tools_used"] = calls
    return update


# ---------------------------------------------------------------- 1. understand

def conversation_text(state: CallState) -> str:
    who = {"client": "Client", "advisor": "Conseiller"}
    return "\n".join(f"{who[m.role]} : {m.content}" for m in state.messages)


def understand(model, state: CallState) -> Understanding:
    prompt = prompts.UNDERSTAND.format(conversation=conversation_text(state))
    try:
        result = model.with_structured_output(Understanding).invoke(prompt)
        if isinstance(result, Understanding):
            return result
        return Understanding.model_validate(result)
    except Exception:
        # small models sometimes answer in text: look for the JSON in it
        reply = model.invoke(prompt + '\n\nRéponds seulement en JSON : {"intent": "...", "is_complaint": true/false}')
        return parse_understanding(reply.content, state)


def parse_understanding(text: str, state: CallState) -> Understanding:
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            return Understanding.model_validate(json.loads(match.group(0)))
        except ValueError:
            pass
    return Understanding(intent=state.intent or "unknown", is_complaint=False)


# ---------------------------------------------------------------- 3. answer

def history(state: CallState) -> list:
    """The call so far, as chat messages: the customer is the human, the advisor is the AI."""
    return [HumanMessage(m.content) if m.role == "client" else AIMessage(m.content) for m in state.messages]


def system_prompt(state: CallState) -> str:
    text = prompts.SYSTEM.format(customer_id=state.customer_id)
    result = state.complaint_result
    if result is not None:
        escalation = ""
        if result.escalate:
            escalation = prompts.ESCALATION.format(reasons=", ".join(result.escalation_reasons) or "règle interne")
        text += "\n\n" + prompts.COMPLAINT_RESULT.format(
            category=result.category,
            severity=result.severity,
            actions="; ".join(result.actions) or "aucune",
            compensation=euros(result.compensation_cents) if result.compensation_cents else "aucune",
            rules=", ".join(result.rules_cited) or "aucune",
            escalation=escalation,
        )
    return text


def answer(model, state: CallState) -> tuple[str, list[ToolCall]]:
    tools = {tool.name: tool for tool in build_tools(state.customer_id)}
    with_tools = model.bind_tools(list(tools.values()))
    messages = [SystemMessage(system_prompt(state)), *history(state)]
    calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = with_tools.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return clean(reply.content), calls
        for call in reply.tool_calls:
            calls.append(ToolCall(name=call["name"], arguments=call.get("args") or {}))
            messages.append(ToolMessage(run_tool(tools, call), tool_call_id=call.get("id") or call["name"]))

    # too many searches: answer now with what was found
    reply = model.invoke(messages + [HumanMessage(prompts.FORCE_ANSWER)])
    return clean(reply.content), calls


def run_tool(tools: dict, call: dict) -> str:
    tool = tools.get(call["name"])
    if tool is None:
        return f"Erreur : l'outil {call['name']} n'existe pas. Outils disponibles : {', '.join(tools)}."
    try:
        return str(tool.invoke(call.get("args") or {}))
    except Exception as error:      # a tool problem must not stop the call
        return f"Erreur : l'outil {call['name']} n'a pas fonctionné ({error})."


def clean(text) -> str:
    if isinstance(text, list):      # some models answer in parts
        text = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    text = (text or "").strip()
    return text or FALLBACK_ANSWER
