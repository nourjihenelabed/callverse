"""Checks the pretend customer (callverse/agents/client).

No Ollama and no PostgreSQL: a scripted fake model plays the customer's choices,
and a temporary SQLite file replaces the database.
"""

import random
from collections import Counter

import pytest
from sqlalchemy import create_engine

from callverse.agents.client import agent, scenarios
from callverse.agents.client.personas import PERSONAS
from callverse.graph import run_call
from callverse.state import CallState, ComplaintResult, Message, Persona
from fakes import FAKE_FACTS, StructuredScript
from test_seed import fake_rows, seed


@pytest.fixture
def database(tmp_path, monkeypatch):
    rows, products = seed.build_customers(fake_rows())
    complaint_rows, ticket_rows = seed.build_complaints(rows, seed=1)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    seed.write_all(engine, rows, products, complaint_rows, ticket_rows)
    monkeypatch.setattr(scenarios, "get_engine", lambda: engine)


def use(monkeypatch, *answers):
    model = StructuredScript(answers)
    monkeypatch.setattr(agent, "get_model", lambda: model)
    return model


def state(persona="calm", reason="check_invoice", turn=1, **fields) -> CallState:
    messages = [Message(role="client", content="Bonjour, j'ai une question."),
                Message(role="advisor", content="Bien sûr, je vous écoute.")][: 2 if turn else 0]
    return CallState(call_id="c", customer_id="0001-AAAAA", persona=persona, call_reason=reason,
                     messages=messages, turn=turn, **fields)


# ---------------------------------------------------------------- moods and reasons

def test_every_mood_of_state_py_is_described():
    assert set(PERSONAS) == set(Persona.__args__)
    assert all(mood.patience >= 3 for mood in PERSONAS.values())


def test_every_reason_brief_can_be_filled():
    for name, reason in scenarios.REASONS.items():
        assert reason.queue in scenarios.QUEUE_SHARE
        text = scenarios.brief(name, FAKE_FACTS)
        assert "{" not in text and len(text) > 30
        assert name.startswith("complaint") == ("réclamation" in text)


def test_customer_knows_their_own_account(database):
    facts = scenarios.customer_facts("0001-AAAAA")                 # fibre, mobile, extra line, security, TV
    assert facts["bill"] == "71,95 €" and facts["double_bill"] == "143,90 €"
    assert facts["box"] == "Box Fibre"
    assert facts["other_offer"] not in facts["offers"]             # an offer they don't have yet
    with pytest.raises(ValueError):
        scenarios.customer_facts("9999-ZZZZZ")


def test_reasons_fit_the_customer_products():
    rng = random.Random(1)
    no_mobile = {**FAKE_FACTS, "codes": ["box_adsl"]}
    no_box = {**FAKE_FACTS, "codes": ["forfait_mobile"]}
    assert all(scenarios.REASONS[scenarios.pick_reason(rng, no_mobile)].needs != "mobile" for _ in range(500))
    assert all(scenarios.REASONS[scenarios.pick_reason(rng, no_box)].needs != "box" for _ in range(500))


def test_queues_follow_the_workforce_call_mix():
    from workforce.simulation import CenterConfig

    assert scenarios.QUEUE_SHARE == CenterConfig().call_mix
    rng = random.Random(2)
    queues = Counter(scenarios.REASONS[scenarios.pick_reason(rng, FAKE_FACTS)].queue for _ in range(4000))
    for queue, share in scenarios.QUEUE_SHARE.items():
        assert abs(queues[queue] / 4000 - share) < 0.03


def test_customers_at_risk_often_call_as_likely_to_churn():
    rng = random.Random(3)
    risky = Counter(scenarios.pick_persona(rng, "general", 0.9) for _ in range(2000))
    safe = Counter(scenarios.pick_persona(rng, "general", 0.1) for _ in range(2000))
    assert risky["likely_to_churn"] > 3 * safe["likely_to_churn"]


def test_random_call_is_a_valid_new_call(database):
    rng = random.Random(4)
    call = scenarios.random_call(rng, "call-1")
    assert call.customer_id in {"0001-AAAAA", "0002-BBBBB", "0003-CCCCC"}
    assert call.call_reason in scenarios.REASONS and call.messages == [] and call.turn == 0
    calls = [scenarios.random_call(rng, f"c{i}", customer_id="0002-BBBBB") for i in range(200)]
    assert all(scenarios.REASONS[c.call_reason].needs != "mobile" for c in calls)   # 0002-BBBBB has no mobile


# ---------------------------------------------------------------- one customer turn

def test_first_turn_explains_the_reason_with_the_account_facts(monkeypatch):
    model = use(monkeypatch, {"message": "« Bonjour, ma facture est de 99,96 € au lieu de 49,98 € ! »"})
    update = agent.client_node(state(persona="dissatisfied", reason="complaint_billing", turn=0))

    assert update == {"messages": [Message(role="client", content="Bonjour, ma facture est de 99,96 € au lieu de 49,98 € !")],
                      "turn": 1}
    prompt = model.sent[0]
    assert "99,96 €" in prompt and "49,98 €" in prompt                 # the situation, with real numbers
    assert PERSONAS["dissatisfied"].description in prompt
    assert "churn" not in prompt.lower()


def test_customer_continues_while_not_answered(monkeypatch):
    use(monkeypatch, {"decision": "continue", "message": "Et ça coûte combien ?", "satisfaction": 3})
    update = agent.client_node(state())
    assert update["turn"] == 2 and "outcome" not in update
    assert update["messages"][0].content == "Et ça coûte combien ?"


@pytest.mark.parametrize("decision, outcome", [("accept", "resolved"), ("refuse", "refused"), ("hang_up", "abandoned")])
def test_customer_ends_the_call(monkeypatch, decision, outcome):
    use(monkeypatch, {"decision": decision, "message": "Au revoir.", "satisfaction": 4})
    update = agent.client_node(state())
    assert update["outcome"] == outcome
    assert update["satisfaction"] == (2 if outcome == "abandoned" else 4)    # hanging up is never a happy call


def test_accepting_an_escalated_complaint_ends_as_escalated(monkeypatch):
    use(monkeypatch, {"decision": "accept", "message": "D'accord, j'attends l'appel.", "satisfaction": 3})
    escalated = ComplaintResult(category="billing", severity=2, actions=[], escalate=True)
    update = agent.client_node(state(reason="complaint_billing", is_complaint=True, complaint_result=escalated))
    assert update["outcome"] == "escalated"


def test_impatient_customer_hangs_up_when_patience_runs_out(monkeypatch):
    use(monkeypatch, {"decision": "continue", "message": "Et donc ?", "satisfaction": 3})
    patience = PERSONAS["impatient"].patience
    update = agent.client_node(state(persona="impatient", turn=patience))
    assert update["outcome"] == "abandoned" and update["satisfaction"] <= 2
    assert update["messages"][0].content == PERSONAS["impatient"].hang_up


def test_the_advisor_is_the_other_person_in_the_conversation(monkeypatch):
    model = use(monkeypatch, {"decision": "continue", "message": "Merci.", "satisfaction": 3})
    agent.client_node(state())
    sent = model.sent[0]
    assert type(sent[1]).__name__ == "AIMessage" and sent[1].content.startswith("Bonjour")     # the customer
    assert type(sent[2]).__name__ == "HumanMessage" and sent[2].content.startswith("Bien sûr")  # the advisor


def test_model_failure_does_not_stop_the_call(monkeypatch):
    use(monkeypatch, RuntimeError("Ollama is down"), RuntimeError("Ollama is down"))
    first = agent.client_node(state(reason="lost_sim", turn=0))
    assert first["messages"][0].content == "Bonjour, je vous appelle pour lost sim."
    last = agent.client_node(state())
    assert last["outcome"] == "resolved" and last["satisfaction"] == 3


def test_no_database_still_gives_a_customer(monkeypatch):
    def broken(customer_id):
        raise RuntimeError("no database")

    monkeypatch.setattr(agent, "get_facts", broken)
    use(monkeypatch, {"message": "Bonjour, ma box ne marche plus."})
    assert agent.client_node(state(reason="internet_outage", turn=0))["turn"] == 1


def test_whole_call_with_the_pretend_customer(monkeypatch):
    use(monkeypatch,
        {"message": "Bonjour, je voudrais comprendre ma facture."},
        {"decision": "continue", "message": "Et l'option TV, c'est combien ?", "satisfaction": 3},
        {"decision": "accept", "message": "Parfait, merci, au revoir.", "satisfaction": 5})
    done = run_call(CallState(call_id="w", customer_id="0001-AAAAA", persona="interested_in_offer",
                              call_reason="check_invoice"))
    assert [m.role for m in done.messages] == ["client", "advisor", "client", "advisor", "client"]
    assert done.outcome == "resolved" and done.satisfaction == 5 and done.turn == 3
    assert done.quality_report is not None
