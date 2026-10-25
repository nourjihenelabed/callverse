"""Pretend customer: plays the person who calls, with one of six moods.

How one customer turn works:
1. First turn: the AI model writes the opening sentence from the mood (personas.py)
   and the situation (scenarios.py), filled with the customer's real account facts.
2. Later turns: the model reads the advisor's answer and decides: continue, accept,
   refuse or hang up, with a satisfaction score from 1 to 5.
3. Code keeps the call realistic: a customer whose patience runs out hangs up, an accepted
   escalated complaint ends as "escalated", and the satisfaction stays between 1 and 5.

Input: the CallState. Output: a dict with the fields it changes:
- "messages": a list with the customer's new Message
- "turn": state.turn + 1
- at the end of the call, also "outcome" and "satisfaction"

The AI model comes from get_model() and the account from get_facts(), so tests can replace them.
"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from callverse.agents.client import prompts
from callverse.agents.client.personas import PERSONAS
from callverse.agents.client.scenarios import brief, customer_facts
from callverse.llm import get_llm
from callverse.state import CallState, Message

log = logging.getLogger(__name__)

TEMPERATURE = 0.7          # a bit of variety: two calls with the same mood should not be identical
OUTCOMES = {"accept": "resolved", "refuse": "refused", "hang_up": "abandoned"}
DEFAULT_SATISFACTION = {"resolved": 4, "escalated": 3, "refused": 2, "abandoned": 1}


class Opening(BaseModel):
    message: str = Field(description="la première phrase du client")


class ClientTurn(BaseModel):
    decision: Literal["continue", "accept", "refuse", "hang_up"]
    message: str = Field(description="ce que dit le client")
    satisfaction: int = Field(ge=1, le=5, description="satisfaction du client, de 1 à 5")


def get_model():
    """The AI model. Tests replace this function with a fake model."""
    return get_llm(temperature=TEMPERATURE)


def get_facts(customer_id: str) -> dict:
    """The customer's account. Tests replace this function."""
    return customer_facts(customer_id)


def client_node(state: CallState) -> dict:
    model = get_model()
    role = role_text(state)

    # 1. first turn: say why you are calling
    if not state.messages:
        text = opening(model, role, state)
        return {"messages": [Message(role="client", content=text)], "turn": state.turn + 1}

    # 2. answer the advisor, or end the call
    turn = reply(model, role, state)
    update = {"messages": [Message(role="client", content=turn.message)], "turn": state.turn + 1}

    # 3. rules kept by code
    decision = turn.decision
    if decision == "continue" and state.turn >= PERSONAS[state.persona].patience:
        decision = "hang_up"
        update["messages"] = [Message(role="client", content=PERSONAS[state.persona].hang_up)]
    if decision != "continue":
        outcome = OUTCOMES[decision]
        if outcome == "resolved" and state.complaint_result is not None and state.complaint_result.escalate:
            outcome = "escalated"           # the customer accepts to wait for the manager
        satisfaction = turn.satisfaction or DEFAULT_SATISFACTION[outcome]
        if outcome == "abandoned":
            satisfaction = min(satisfaction, 2)
        update |= {"outcome": outcome, "satisfaction": max(1, min(5, satisfaction))}
    return update


# ---------------------------------------------------------------- the model's parts

def role_text(state: CallState) -> str:
    try:
        facts = get_facts(state.customer_id)
    except Exception as error:      # no database: a vaguer customer, but the call goes on
        log.warning("client: account not found (%s)", error)
        facts = {"offers": "inconnues", "bill": "inconnue", "contract": "inconnu", "payment": "inconnu",
                 "tenure": "?", "double_bill": "le double", "box": "votre box", "other_offer": "la Box Fibre",
                 "other_option": "Cinéma & Séries"}
    mood = PERSONAS[state.persona].description
    return prompts.ROLE.format(mood=mood, brief=brief(state.call_reason, facts), **facts)


def conversation(state: CallState) -> list:
    """The call from the customer's side: the advisor is the other person, the customer is the AI."""
    return [AIMessage(m.content) if m.role == "client" else HumanMessage(m.content) for m in state.messages]


def opening(model, role: str, state: CallState) -> str:
    prompt = prompts.OPENING.format(role=role)
    try:
        return clean(model.with_structured_output(Opening).invoke(prompt).message) or fallback_opening(state)
    except Exception as error:
        log.warning("client: opening failed, using a fixed sentence (%s)", error)
        return fallback_opening(state)


def reply(model, role: str, state: CallState) -> ClientTurn:
    messages = [SystemMessage(prompts.REPLY.format(role=role)), *conversation(state)]
    try:
        turn = model.with_structured_output(ClientTurn).invoke(messages)
        turn = turn if isinstance(turn, ClientTurn) else ClientTurn.model_validate(turn)
        if clean(turn.message):
            return turn.model_copy(update={"message": clean(turn.message)})
    except Exception as error:
        log.warning("client: reply failed, ending politely (%s)", error)
    return ClientTurn(decision="accept", message="D'accord, merci. Au revoir.", satisfaction=3)


def fallback_opening(state: CallState) -> str:
    return f"Bonjour, je vous appelle pour {state.call_reason.replace('complaint_', 'une réclamation ').replace('_', ' ')}."


def clean(text: str) -> str:
    """No quotes or role labels around the sentence ("Client : ...")."""
    text = (text or "").strip().strip('"«»“” ').strip()
    for label in ("Client :", "Client:", "CLIENT :", "Moi :"):
        if text.startswith(label):
            text = text[len(label):].strip()
    return text
