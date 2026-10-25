"""Advisor: answers the call and asks the complaint specialist for help when needed.

How one advisor turn works:
1. Understand: while no complaint has been handled, the AI model reads the customer's
   last message and decides the intent and whether it is a complaint.
2. A complaint the specialist has not seen yet: return {"is_complaint": True} with no
   message. The graph sends the call to the specialist, then back here.
3. Answer: first the code reads the customer's file and searches the rules with the
   customer's last message (prefetch). Then the AI model answers, and may still use its
   tools (search_policies, get_complaint_history) up to MAX_TOOL_ROUNDS times. When the specialist has decided, its decision is given to the
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
from callverse.agents.advisor.tools import build_tools, customer_file, euros, policy_search, product_codes
from callverse.llm import get_llm
from callverse.state import CallState, Message, ToolCall

MAX_TOOL_ROUNDS = 4
MIN_QUESTION_WORDS = 3
# words in the customer's message -> the product it needs (the product codes of catalogue.md)
NEEDS_PRODUCT = [
    (re.compile(r"\bsim\b|ligne mobile|forfait mobile|portable|\bmobile\b", re.I), "forfait_mobile", "ligne mobile"),
    (re.compile(r"\bbox\b|internet|fibre|adsl|wifi|wi-fi", re.I), "box_", "box internet"),
]
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


def last_client_message(state: CallState) -> str:
    return next((m.content for m in reversed(state.messages) if m.role == "client"), "")


def prefetch(state: CallState) -> tuple[str, list[ToolCall]]:
    """Before the model answers, the code itself reads the customer's file and searches the
    rules with the customer's own words. A small model often forgets a tool or writes a bad
    search, so it starts with what it needs; it can still search again with search_policies."""
    question = last_client_message(state)
    try:
        customer = customer_file(state.customer_id)
    except Exception as error:      # no database: the call goes on
        customer = f"Dossier indisponible ({error})."
    try:
        rules = policy_search(question)
    except Exception as error:      # no rules index: the call goes on
        rules = f"Recherche indisponible ({error})."
    calls = [ToolCall(name="get_customer", arguments={}),
             ToolCall(name="search_policies", arguments={"question": question})]
    return prompts.CONTEXT.format(customer=customer, rules=rules, warnings=missing_products(state, question)), calls


def missing_products(state: CallState, question: str) -> str:
    """A warning when the customer asks about a product they do not have (checked by code, not the model).
    Not for offer questions ("combien coûte la Box Fibre ?"): anyone may ask about an offer."""
    if re.search(r"co[uû]te|prix|tarif|souscrire|offre", question, re.I):
        return ""
    try:
        codes = product_codes(state.customer_id)
    except Exception:               # no database: no warning
        return ""
    return "".join(prompts.NO_PRODUCT.format(product=label) for pattern, code, label in NEEDS_PRODUCT
                   if pattern.search(question) and not any(c.startswith(code) for c in codes))


def answer(model, state: CallState) -> tuple[str, list[ToolCall]]:
    tools = {tool.name: tool for tool in build_tools(state.customer_id)}
    with_tools = model.bind_tools(list(tools.values()))
    context, calls = prefetch(state)
    messages = [SystemMessage(system_prompt(state) + context), *history(state)]

    for _ in range(MAX_TOOL_ROUNDS):
        reply = text_tool_call(with_tools.invoke(messages), tools)
        messages.append(reply)
        if not reply.tool_calls:
            return clean(reply.content), calls
        for call in reply.tool_calls:
            args = fix_arguments(call, state)
            calls.append(ToolCall(name=call["name"], arguments=args))
            messages.append(ToolMessage(run_tool(tools, call["name"], args), tool_call_id=call.get("id") or call["name"]))

    # too many searches: answer now with what was found
    reply = model.invoke(messages + [HumanMessage(prompts.FORCE_ANSWER)])
    return clean(reply.content), calls


def text_tool_call(reply: AIMessage, tools: dict) -> AIMessage:
    """Llama 3.1 sometimes writes its tool call as text: {"name": "search_policies", "parameters": {...}}.
    Turn that text into a real tool call, so the customer never hears JSON."""
    text = reply.content if isinstance(reply.content, str) else ""
    if reply.tool_calls or not text.strip().startswith("{"):
        return reply
    try:
        data = json.loads(text.strip())
    except ValueError:
        return reply
    if not isinstance(data, dict) or data.get("name") not in tools:
        return reply
    args = data.get("parameters") or data.get("arguments") or {}
    return AIMessage(content="", tool_calls=[{"name": data["name"], "args": args, "id": f"text-{data['name']}"}])


def fix_arguments(call: dict, state: CallState) -> dict:
    """A search cut short by the model (for example 'J' from "J'ai perdu...") uses the customer's words."""
    args = dict(call.get("args") or {})
    if call["name"] == "search_policies" and len(str(args.get("question", "")).split()) < MIN_QUESTION_WORDS:
        args["question"] = last_client_message(state)
    return args


def run_tool(tools: dict, name: str, args: dict) -> str:
    tool = tools.get(name)
    if tool is None:
        return f"Erreur : l'outil {name} n'existe pas. Outils disponibles : {', '.join(tools)}."
    try:
        return str(tool.invoke(args))
    except Exception as error:      # a tool problem must not stop the call
        return f"Erreur : l'outil {name} n'a pas fonctionné ({error})."


def clean(text) -> str:
    if isinstance(text, list):      # some models answer in parts
        text = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    text = (text or "").strip()
    return text or FALLBACK_ANSWER
