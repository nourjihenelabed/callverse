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
from difflib import SequenceMatcher

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from callverse.agents.advisor import prompts
from callverse.agents.advisor.tools import (build_tools, catalogue, customer_file, euros, policy_search,
                                            product_codes)
from callverse.llm import get_llm
from callverse.state import CallState, Message, ToolCall

MAX_TOOL_ROUNDS = 4
MIN_QUESTION_WORDS = 3
REPEAT_SIMILARITY = 0.8    # 80% the same words in the same order = the same answer again
MAX_SENTENCES = 5           # the prompt asks for 4; the code cuts what is beyond 5
COMPENSATION_WORDS = re.compile(r"compensation|geste commercial|\bun avoir\b|dédommag|indemni|offrir|offert", re.I)
ESCALATION_ACTION = re.compile(r"transmis à un responsable", re.I)   # rules.ESCALATION_ACTION, said by the reminder
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
    wants_to_leave: bool = Field(default=False, description="true if the customer wants to cancel or leave")


def get_model():
    """The AI model. Tests replace this function with a fake model."""
    return get_llm()


def advisor_node(state: CallState) -> dict:
    model = get_model()
    update = {}

    # 1. understand the customer's last message (until a complaint and a departure have been handled)
    if state.complaint_result is None or state.retention is None:
        understood = understand(model, state)
        if state.complaint_result is None:
            if state.intent is None or (understood.is_complaint and not state.is_complaint):
                update["intent"] = understood.intent
            update["is_complaint"] = state.is_complaint or understood.is_complaint
        if not state.wants_to_leave and (understood.wants_to_leave or leaving(state)):
            update["wants_to_leave"] = True

        # 2. a complaint: the specialist decides first; a customer who wants to leave: the supervisor
        if update.get("is_complaint") and state.complaint_result is None:
            return update
        if (state.wants_to_leave or update.get("wants_to_leave")) and state.retention is None:
            return update

    # 3. answer the customer, with the tools
    text, calls = answer(model, state)
    update["messages"] = [Message(role="advisor", content=text)]
    if calls:
        update["tools_used"] = calls
    return update


# ---------------------------------------------------------------- 1. understand

LEAVING = re.compile(r"(veux|voudrais|vais|souhaite|aimerais|compte|envisage de) (vous )?(résilier|quitter|partir)|"
                     r"résilier mon|quitter callverse|changer d'opérateur|partir chez|(aller|voir) ailleurs|"
                     r"code rio|chez la concurrence|un autre opérateur", re.I)


def leaving(state: CallState) -> bool:
    """A departure said in plain words (the small model sometimes misses it)."""
    return bool(LEAVING.search(last_client_message(state)))


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


def catalogue_text() -> str:
    """The 10 offers and their prices, always given to the model (it once said the Box ADSL did not exist)."""
    try:
        return "\n".join(f"- {name} : {euros(cents)}/mois" for name, cents in catalogue().values())
    except OSError:
        return "(catalogue indisponible : utilise search_policies)"


def system_prompt(state: CallState) -> str:
    text = prompts.SYSTEM.format(customer_id=state.customer_id, catalogue=catalogue_text())
    result = state.complaint_result
    if result is not None:
        actions = [a for a in result.actions if not ESCALATION_ACTION.search(a)]
        text += "\n\n" + prompts.COMPLAINT_RESULT.format(
            category=result.category,
            severity=result.severity,
            actions="; ".join(actions) or "aucune",
            compensation=euros(result.compensation_cents) if result.compensation_cents else "aucune",
            rules=", ".join(result.rules_cited) or "aucune",
            escalation=prompts.ESCALATION if result.escalate else "",
        )
    return text


def retention_text(state: CallState) -> str:
    r = state.retention
    if r is None:
        return ""
    if r.supervisor and r.offer:
        return prompts.RETENTION_OFFER.format(offer=r.offer)
    if r.supervisor:
        return prompts.RETENTION_CALLBACK
    return prompts.RETENTION_CONDITIONS


def reminder(state: CallState) -> str:
    """The last instruction before the model answers, chosen by the situation."""
    if state.retention is not None and state.complaint_result is None:
        return prompts.RETENTION_REMINDER
    result = state.complaint_result
    if result is None:
        return prompts.REMINDER
    if result.escalate:
        offer = state.retention.offer if state.retention and state.retention.offer else None
        return prompts.ESCALATED_REMINDER + (prompts.OFFER_STILL_VALID.format(offer=offer) if offer else "")
    return prompts.COMPLAINT_REMINDER.format(compensation=compensation_text(result.compensation_cents))


def compensation_text(cents: int) -> str:
    return euros(cents) if cents else "aucune"


# ---------------------------------------------------------------- checks on the answer, done by code

def sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def keep_compensation(text: str, state: CallState) -> str:
    """The advisor may not invent or change the specialist's compensation (it once promised 20 € instead of
    11,70 €). A sentence about compensation with another amount is replaced by the decided amount."""
    result = state.complaint_result
    if result is None:
        return text
    decided = result.compensation_cents
    fixed, said = [], False
    for sentence in sentences(text):
        amounts = {round(float(a.replace(",", ".")) * 100) for a in re.findall(r"(\d+(?:,\d{1,2})?)\s*(?:€|euros?)", sentence)}
        about_money = COMPENSATION_WORDS.search(sentence)
        if about_money and (amounts - {decided} or (amounts and result.escalate)):
            if not said and not result.escalate:
                fixed.append(prompts.COMPENSATION_SENTENCE.format(amount=euros(decided)) if decided
                             else prompts.NO_COMPENSATION_SENTENCE)
                said = True
            continue                        # the invented amount is dropped
        fixed.append(sentence)
    return " ".join(fixed)


def tidy_answer(text: str) -> str:
    """At most MAX_SENTENCES sentences, and one question at the end, not two."""
    parts = sentences(text)
    closing = parts[-1] if parts and parts[-1] == prompts.CLOSING else None
    body = [p for p in parts if p != prompts.CLOSING]
    if any(p.endswith("?") for p in body):
        closing = None                      # it already asked a question: one at a time
    body = body[:MAX_SENTENCES]
    return " ".join(body + ([closing] if closing else [])) or FALLBACK_ANSWER


def tell_escalation(text: str, state: CallState) -> str:
    """An escalated complaint: the customer must hear it, even if the model forgot to say it."""
    result = state.complaint_result
    if result is None or not result.escalate or re.search(r"responsable|superviseur|48 ?h", text, re.I):
        return text
    if text.endswith(prompts.CLOSING):
        return f"{text[:-len(prompts.CLOSING)].rstrip()} {prompts.ESCALATION_SENTENCE} {prompts.CLOSING}"
    return f"{text} {prompts.ESCALATION_SENTENCE}"


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
    text, calls, messages = first_answer(model, state)
    if repeats_itself(text, state):
        # the small model sometimes says the same thing turn after turn: ask once for a new answer
        retry = model.invoke(messages + [HumanMessage(prompts.DONT_REPEAT)])
        text = finish(clean(retry.content), state)
        if repeats_itself(text, state) or looks_like_tool_call(text):
            text = prompts.NOTHING_MORE
    return text, calls


def repeats_itself(text: str, state: CallState) -> bool:
    """True when the answer is (almost) one the advisor already gave in this call."""
    body = text.replace(prompts.CLOSING, "").strip()
    earlier = [m.content.replace(prompts.CLOSING, "").strip() for m in state.messages if m.role == "advisor"]
    return any(SequenceMatcher(None, body, old).ratio() >= REPEAT_SIMILARITY for old in earlier)


def first_answer(model, state: CallState) -> tuple[str, list[ToolCall], list]:
    tools = {tool.name: tool for tool in build_tools(state.customer_id)}
    with_tools = model.bind_tools(list(tools.values()))
    context, calls = prefetch(state)
    messages = [SystemMessage(system_prompt(state) + retention_text(state) + context + reminder(state)), *history(state)]

    for _ in range(MAX_TOOL_ROUNDS):
        reply = text_tool_call(with_tools.invoke(messages), tools)
        messages.append(reply)
        if not reply.tool_calls:
            return finish(clean(reply.content), state), calls, messages
        for call in reply.tool_calls:
            args = fix_arguments(call, state)
            calls.append(ToolCall(name=call["name"], arguments=args))
            messages.append(ToolMessage(run_tool(tools, call["name"], args), tool_call_id=call.get("id") or call["name"]))

    # too many searches: answer now with what was found
    reply = model.invoke(messages + [HumanMessage(prompts.FORCE_ANSWER)])
    text = clean(reply.content)
    if looks_like_tool_call(text):          # still JSON: never say it to the customer
        text = FALLBACK_ANSWER
    return finish(text, state), calls, messages


def finish(text: str, state: CallState) -> str:
    """The checks done by code on every answer, in order."""
    text = no_consent_without_request(keep_compensation(text, state), state)
    text = keep_retention_offer(tidy_answer(text), state)
    return tell_escalation(text, state)


NO_OFFER = re.compile(r"(pas|aucune?|plus) d[e'’ ]\s*(offre|avantage|geste|réduction|remise|promotion)|"
                      r"rien à vous proposer", re.I)


def keep_retention_offer(text: str, state: CallState) -> str:
    """The supervisor's loyalty offer must reach the customer, and never be contradicted.
    (The model once explained the conditions without the offer, then said "nous n'avons pas d'offre".)"""
    r = state.retention
    if r is None or not r.offer:
        return text
    kept = [s for s in sentences(text) if not NO_OFFER.search(s)]       # no "we have no offer" next to an offer
    text = " ".join(kept) or text
    said_before = any(r.offer_key.lower() in m.content.lower() for m in state.messages if m.role == "advisor")
    if said_before or r.offer_key.lower() in text.lower():
        return text
    sentence = prompts.RETENTION_SENTENCE.format(offer=r.offer)
    if text.endswith(prompts.CLOSING):
        return f"{text[:-len(prompts.CLOSING)].rstrip()} {sentence} {prompts.CLOSING}"
    return f"{text} {sentence}"


CHANGE_REQUEST = re.compile(r"changer|modifier|ajouter|souscrire|activer|suspendre|résilier|annuler|passer à|"
                            r"prendre|commander|je veux|je voudrais (le|la|l')|oui|d'accord|allez-y", re.I)


def no_consent_without_request(text: str, state: CallState) -> str:
    """"Voulez-vous que je le fasse maintenant ?" only when the customer asked for a change
    (the model once asked it right after the customer said who they were)."""
    if prompts.ASK_CONSENT not in text or CHANGE_REQUEST.search(last_client_message(state)):
        return text
    kept = [s for s in sentences(text) if prompts.ASK_CONSENT not in s]
    return " ".join(kept + [prompts.CLOSING])


def looks_like_tool_call(text: str) -> bool:
    return bool(re.match(r'\s*\{\s*"name"\s*:', text or ""))


def text_tool_call(reply: AIMessage, tools: dict) -> AIMessage:
    """Llama 3.1 sometimes writes its tool call as text: {"name": "search_policies", "parameters": {...}}.
    Turn that text into a real tool call, so the customer never hears JSON."""
    text = reply.content if isinstance(reply.content, str) else ""
    if reply.tool_calls or not looks_like_tool_call(text):
        return reply
    try:
        data = json.loads(text.strip().replace("\\'", "'"))     # it writes j\'ai, which is not valid JSON
    except ValueError:
        data = {}
    if not isinstance(data, dict) or "name" not in data:      # still broken: read the two fields directly
        name = re.search(r'"name"\s*:\s*"(\w+)"', text)
        question = re.search(r'"question"\s*:\s*"(.*?)"\s*\}', text, re.S)
        data = {"name": name.group(1) if name else "",
                "parameters": {"question": question.group(1).replace("\\'", "'")} if question else {}}
    if data.get("name") not in tools:
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
    text = re.sub(r",?\s*(je m'appelle|mon nom est|je suis|ici)\s*\[[^\]]*\]", "", text or "", flags=re.I)
    text = re.sub(r"\s*\[[^\]]*\]", "", text)              # placeholders: "[Nom]"
    text = re.sub(r"\s*\((pause|silence|un temps|[^)]*vérifi[^)]*)\)", "", text, flags=re.I)  # "(pause)"
    text = re.sub(r"^\s*,\s*", "", text).strip()
    return (text[:1].upper() + text[1:]) if text else FALLBACK_ANSWER
