"""Checks the retention route: a customer who wants to leave goes to the supervisor once.

No Ollama and no database: fake models and a made-up account (see conftest.py).
"""

import pytest

from callverse.agents.advisor import agent as advisor
from callverse.agents.quality import checks
from callverse.agents.retention import agent as retention
from callverse.graph import run_call
from callverse.state import CallState, Message, RetentionResult
from fakes import ScriptedModel


# ---------------------------------------------------------------- the supervisor's rules

def test_low_risk_gets_the_conditions_and_no_offer():
    r = retention.decide(0.4, {"box_fibre"})
    assert not r.supervisor and r.offer is None


@pytest.mark.parametrize("codes, offer_key, offer_part", [
    ({"box_adsl", "forfait_mobile"}, "Box Fibre", "Box ADSL à la Box Fibre"),     # the free upgrade first
    ({"box_fibre"}, "3 mois offerts", "option TV"),                               # a free option
    ({"box_fibre", "option_tv"}, "3 mois offerts", "Cinéma & Séries"),            # one they do not have yet
])
def test_high_risk_gets_an_offer_from_the_grid(codes, offer_key, offer_part):
    r = retention.decide(0.8, codes)
    assert r.supervisor and r.offer_key == offer_key and offer_part in r.offer
    assert "compensation_grid.md > Offres de fidélisation" in r.rules_cited


def test_mobile_only_high_risk_goes_to_the_supervisor_without_an_offer():
    r = retention.decide(0.9, {"forfait_mobile"})
    assert r.supervisor and r.offer is None


def test_threshold_is_the_complaint_rules_one():
    assert retention.decide(0.69, {"box_fibre"}).supervisor is False
    assert retention.decide(0.70, {"box_fibre"}).supervisor is True


# ---------------------------------------------------------------- the advisor

def call(*client_lines, **fields) -> CallState:
    return CallState(call_id="r", customer_id="0001-AAAAA", persona="likely_to_churn", call_reason="cancel_contract",
                     messages=[Message(role="client", content=t) for t in client_lines], **fields)


@pytest.mark.parametrize("line, leaving", [
    ("Je veux résilier mon abonnement.", True),
    ("Je vais aller voir ailleurs, la concurrence est moins chère.", True),
    ("Comment obtenir mon code RIO ?", True),
    ("Je conteste les frais de résiliation de mon ancien contrat.", False),
    ("Combien coûte la Box Fibre ?", False),
])
def test_departure_words(line, leaving):
    assert advisor.leaving(call(line)) is leaving


def test_advisor_sends_a_departure_to_the_supervisor_once(monkeypatch):
    monkeypatch.setattr(advisor, "get_model", lambda: ScriptedModel(intent="cancel_contract"))
    update = advisor.advisor_node(call("Je veux résilier mon abonnement."))
    assert update.get("wants_to_leave") is True and "messages" not in update


def test_advisor_presents_the_supervisor_offer_without_the_score(monkeypatch):
    model = ScriptedModel(["Un responsable vous propose 3 mois offerts sur l'option TV."])
    monkeypatch.setattr(advisor, "get_model", lambda: model)
    decided = retention.decide(0.85, {"box_fibre"})
    update = advisor.advisor_node(call("Je veux partir.", wants_to_leave=True, retention=decided))
    prompt = next(sent for sent in model.sent if isinstance(sent, list))[0].content
    assert "3 mois offerts sur l'option TV" in prompt and "0.85" not in prompt
    assert update["messages"][0].content.startswith("Un responsable")


def test_whole_call_goes_through_the_supervisor(monkeypatch):
    monkeypatch.setattr(retention, "get_facts", lambda customer_id: {"churn_risk": 0.9, "codes": {"box_adsl"}})
    done = run_call(call().model_copy(update={"messages": []}))
    # the fake customer opens with "comprendre ma dernière facture": no departure, no supervisor
    assert done.retention is None

    class Leaving(ScriptedModel):
        def with_structured_output(self, schema):
            outer = self

            class S:
                def invoke(self, prompt):
                    if schema.__name__ == "Opening":
                        return schema(message="Bonjour, je veux résilier mon abonnement, c'est trop cher.")
                    if schema.__name__ == "ClientTurn":
                        return schema(decision="accept", message="D'accord, j'accepte l'offre. Au revoir.", satisfaction=4)
                    return schema(intent="cancel_contract", is_complaint=False, wants_to_leave=True)
            return S()

    from callverse.agents.client import agent as client
    monkeypatch.setattr(client, "get_model", lambda: Leaving())
    monkeypatch.setattr(advisor, "get_model", lambda: Leaving(["Un responsable vous propose le passage gratuit à la Box Fibre."]))
    done = run_call(call().model_copy(update={"messages": []}))
    assert done.wants_to_leave and done.retention.offer_key == "Box Fibre"
    assert [m.role for m in done.messages] == ["client", "advisor", "client"]
    assert done.outcome == "resolved"


# ---------------------------------------------------------------- the reviewer

def test_reviewer_checks_the_offer_was_presented():
    decided = RetentionResult(high_risk=True, supervisor=True, offer="3 mois offerts sur l'option TV",
                              offer_key="3 mois offerts")
    silent = call("Je veux partir.", retention=decided, wants_to_leave=True).model_copy(update={"messages": [
        Message(role="client", content="Je veux partir."), Message(role="advisor", content="Je comprends.")]})
    told = silent.model_copy(update={"messages": [Message(role="client", content="Je veux partir."),
                                                   Message(role="advisor", content="Je vous propose 3 mois offerts.")]})
    assert any("pas été présentée" in f.text for f in checks.compliance_findings(silent))
    assert not any("présentée" in f.text for f in checks.compliance_findings(told))


# ---------------------------------------------------------------- the offer always reaches the customer
# (a real call: the supervisor offered 3 free months on TV, the advisor first gave only the conditions,
# then, after a complaint was escalated, said "nous n'avons pas d'offre ... pour rester fidèle")

OFFER = RetentionResult(high_risk=True, supervisor=True, offer="3 mois offerts sur l'option TV",
                        offer_key="3 mois offerts")


def test_a_forgotten_offer_is_added():
    state = call("Je veux résilier mon abonnement.", wants_to_leave=True, retention=OFFER)
    text = advisor.finish("Vous pouvez résilier sans frais. La résiliation prend effet sous 10 jours. "
                          "Puis-je vous aider pour autre chose ?", state)
    assert "3 mois offerts sur l'option TV" in text and text.endswith("Puis-je vous aider pour autre chose ?")


def test_saying_there_is_no_offer_is_removed():
    from callverse.state import ComplaintResult

    escalated = ComplaintResult(category="service_quality", severity=2, actions=[], escalate=True)
    state = call("Vous n'avez rien à me proposer ?", wants_to_leave=True, retention=OFFER,
                 is_complaint=True, complaint_result=escalated)
    text = advisor.finish("Je comprends votre déception. Malheureusement, nous n'avons pas d'offre de réduction "
                          "de facture ou d'avantage pour rester fidèle à l'heure actuelle. Votre dossier est transmis "
                          "à un responsable qui vous contactera sous 48 heures ouvrées.", state)
    assert "pas d'offre" not in text and "3 mois offerts" in text and "48 heures" in text


def test_the_offer_is_not_repeated_every_turn():
    state = call("Je veux partir.", wants_to_leave=True, retention=OFFER).model_copy(update={"messages": [
        Message(role="client", content="Je veux partir."),
        Message(role="advisor", content="Un responsable vous propose 3 mois offerts sur l'option TV."),
        Message(role="client", content="Et les frais ?")]})
    assert advisor.finish("Il n'y a pas de frais de résiliation.", state) == "Il n'y a pas de frais de résiliation."


def test_escalated_complaint_instructions_keep_the_offer():
    from callverse.state import ComplaintResult

    escalated = ComplaintResult(category="service_quality", severity=2, actions=[], escalate=True)
    state = call("Je suis déçu.", wants_to_leave=True, retention=OFFER, is_complaint=True, complaint_result=escalated)
    assert "reste valable" in advisor.reminder(state) and "3 mois offerts" in advisor.reminder(state)
