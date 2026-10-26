"""Checks the complaint specialist: the 7 steps, the money limits and the supervisor cases.

No Ollama and no database: a fake model gives fixed answers and the history is made up.
"""

import pytest

from callverse.agents.complaint import rules, steps
from callverse.graph import run_call
from callverse.state import CallState, ComplaintResult, Message

GRID_HIT = {"text": "Grille : 5 € en gravité 2.", "source": "compensation_grid.md",
            "section": "Erreur de facturation (catégorie facturation)", "score": 0.9}
PROCEDURE_HIT = {"text": "Vérifier la facture.", "source": "complaint_procedures.md",
                 "section": "Étapes obligatoires par catégorie", "score": 0.8}


class FakeModel:
    """Answers each structured question with a fixed object, like ChatOllama would."""

    def __init__(self, classification=None, draft=None, fail=False):
        self.answers = {steps._Classification: classification, steps._Draft: draft}
        self.fail = fail

    def with_structured_output(self, schema):
        model = self

        class Runner:
            def invoke(self, prompt):
                if model.fail:
                    raise ValueError("the model wrote something that is not JSON")
                return model.answers[schema]

        return Runner()


def call(text="Ma facture a doublé ce mois-ci, ce n'est pas normal.", reason="complaint_billing"):
    return CallState(call_id="t1", customer_id="0013-SMEOE", persona="dissatisfied", call_reason=reason,
                     messages=[Message(role="client", content=text)])


def tools(model=None, history=None, hits=(GRID_HIT, PROCEDURE_HIT)):
    tickets = []

    def open_ticket(state, result, history):
        tickets.append(result)
        return "TCK-000999"

    t = steps.Tools(
        llm=model,
        get_history=lambda customer_id, category: history or steps.History(),
        search=lambda question: list(hits),
        open_ticket=open_ticket,
        update_ticket=lambda ticket_id, new_state: None,
    )
    t.tickets = tickets
    return t


def billing_model(cents=500, severity=2, rules_used=(1,)):
    return FakeModel(
        classification=steps._Classification(category="billing", severity=severity),
        draft=steps._Draft(actions=["rembourser l'option facturée par erreur"],
                           compensation_cents=cents, rules_used=list(rules_used)),
    )


# ---------------------------------------------------------------- normal case

def test_simple_billing_complaint_is_handled_without_a_supervisor():
    t = tools(billing_model(cents=500))
    result = steps.handle_complaint(call(), t)

    assert result.category == "billing" and result.severity == 2
    assert result.compensation_cents == 500
    assert not result.escalate and result.escalation_reasons == []
    assert result.actions[0] == rules.REQUIRED_ACTIONS["billing"]      # the required step comes first
    assert "rembourser l'option facturée par erreur" in result.actions
    assert "compensation_grid.md > Erreur de facturation (catégorie facturation)" in result.rules_cited
    assert result.ticket_id == "TCK-000999" and len(t.tickets) == 1


# ---------------------------------------------------------------- money limits

def test_compensation_above_the_automatic_cap_goes_to_a_supervisor():
    result = steps.handle_complaint(call(), tools(billing_model(cents=1200)))   # gravité 2: auto 5 €, max 15 €

    assert result.compensation_cents == 1200
    assert result.escalate
    assert "compensation au-delà du plafond automatique" in result.escalation_reasons


def test_compensation_is_cut_to_the_supervisor_cap():
    result = steps.handle_complaint(call(), tools(billing_model(cents=10000)))

    assert result.compensation_cents == rules.SUPERVISOR_CAP_CENTS[2]


def test_severity_1_gives_apologies_only():
    result = steps.handle_complaint(call(), tools(billing_model(cents=0, severity=1)))

    assert result.compensation_cents == 0 and not result.escalate


def test_a_compensation_always_cites_the_grid():
    result = steps.handle_complaint(call(), tools(billing_model(cents=300, rules_used=())))

    assert any(r.startswith("compensation_grid.md") for r in result.rules_cited)


def test_made_up_rule_numbers_are_ignored():
    result = steps.handle_complaint(call(), tools(billing_model(rules_used=(1, 42))))

    assert all("42" not in r for r in result.rules_cited)


# ---------------------------------------------------------------- supervisor cases

def test_third_complaint_of_the_same_category_goes_to_a_supervisor():
    result = steps.handle_complaint(call(), tools(billing_model(cents=0), steps.History(repeat_count=2)))

    assert result.escalate and result.repeat_count == 2
    assert result.escalation_reasons[0].startswith("réclamation répétée")
    assert rules.ESCALATION_ACTION in result.actions


def test_the_supervisor_action_is_kept_even_after_a_long_draft():
    model = billing_model(cents=0)
    model.answers[steps._Draft] = steps._Draft(actions=[f"action {n}" for n in range(8)])
    result = steps.handle_complaint(call(), tools(model, steps.History(repeat_count=2)))

    assert result.actions[0] == rules.REQUIRED_ACTIONS["billing"]
    assert result.actions[-1] == rules.ESCALATION_ACTION
    assert len(result.actions) == 2 + steps.MAX_DRAFT_ACTIONS


def test_second_complaint_does_not():
    result = steps.handle_complaint(call(), tools(billing_model(cents=0), steps.History(repeat_count=1)))

    assert not result.escalate


def test_high_churn_risk_goes_to_a_supervisor():
    result = steps.handle_complaint(call(), tools(billing_model(cents=0), steps.History(churn_risk=0.70)))

    assert result.escalation_reasons == ["risque élevé de résiliation"]


def test_legal_threat_is_caught_even_if_the_model_misses_it():
    text = "C'est inadmissible, je vais contacter mon avocat et l'UFC-Que Choisir."
    result = steps.handle_complaint(call(text), tools(billing_model(cents=0)))   # model says legal_threat=False

    assert "menace juridique" in result.escalation_reasons


def test_the_word_proces_is_found_without_accents():
    assert rules.mentions_legal_threat("Je vais faire un PROCÈS")
    assert not rules.mentions_legal_threat("Je voudrais comprendre le processus de facturation")


# ---------------------------------------------------------------- without a working model

def test_no_model_gives_a_safe_answer_from_the_call_reason():
    result = steps.handle_complaint(call(reason="complaint_technical"), tools(model=None))

    assert result.category == "technical"
    assert result.compensation_cents == 0
    assert result.actions == [rules.REQUIRED_ACTIONS["technical"]]
    assert result.rules_cited                                           # still justified by a rule


def test_a_failing_model_does_not_stop_the_call():
    result = steps.handle_complaint(call(reason="complaint_delivery"), tools(FakeModel(fail=True)))

    assert result.category == "delivery" and not result.escalate


# ---------------------------------------------------------------- after the call: the ticket follows the answer

def finished(outcome, escalate=False, ticket_id="TCK-000999"):
    state = call()
    state.outcome = outcome
    state.complaint_result = ComplaintResult(category="billing", severity=2, actions=["x"],
                                             escalate=escalate, ticket_id=ticket_id)
    return state


def recording_tools():
    t = tools()
    t.updates = []
    t.update_ticket = lambda ticket_id, new_state: t.updates.append((ticket_id, new_state))
    return t


def test_accepted_fix_marks_the_ticket_accepted():
    t = recording_tools()
    assert steps.follow_up(finished("resolved"), t) == "accepted"
    assert t.updates == [("TCK-000999", "accepted")]


def test_refused_fix_marks_the_ticket_refused():
    t = recording_tools()
    steps.follow_up(finished("refused"), t)
    assert t.updates == [("TCK-000999", "refused")]


def test_escalated_or_abandoned_ticket_keeps_waiting():
    t = recording_tools()
    steps.follow_up(finished("resolved", escalate=True), t)    # "d'accord" is not the supervisor's decision
    steps.follow_up(finished("abandoned"), t)
    steps.follow_up(finished("resolved", ticket_id=None), t)
    assert t.updates == []


def test_graph_runs_the_follow_up_after_a_complaint_call(monkeypatch):
    t = recording_tools()
    t.llm = None
    monkeypatch.setattr(steps, "make_tools", lambda: t)

    done = run_call(CallState(call_id="g", customer_id="7590-VHVEG", persona="calm", call_reason="complaint_billing"))

    assert done.outcome == "resolved" and done.quality_report is not None
    assert t.updates == [("TCK-000999", "accepted")]


def test_graph_skips_the_follow_up_on_an_everyday_call(monkeypatch):
    t = recording_tools()
    monkeypatch.setattr(steps, "make_tools", lambda: t)

    run_call(CallState(call_id="g", customer_id="7590-VHVEG", persona="calm", call_reason="check_invoice"))

    assert t.updates == []


def test_required_step_is_not_listed_twice_even_reworded():
    case = steps.Case()
    case.category = "billing"
    case.actions = ["Vérifier la facture concernée et les produits du client avant toute proposition.",
                    "Rembourser le montant facturé à tort en totalité.", "rembourser le montant facturé à tort"]
    steps.check_fix(None, case, None)
    assert case.actions == [rules.REQUIRED_ACTIONS["billing"], "Rembourser le montant facturé à tort en totalité."]
    assert steps.RULE_REQUIRED not in case.rules_cited      # the model had already proposed it


def test_different_steps_are_kept():
    assert not rules.same_step("Rembourser le montant", "Proposer un virement")
    assert rules.same_step("vérifier la facture", "Vérifier la facture concernée.")


@pytest.mark.parametrize("category, cents, expected", [
    ("billing", 50, 0),            # 0,50 €: cents and euros mixed up
    ("billing", 480, 500),         # whole euros
    ("billing", 500, 500),
    ("technical", 585, 585),       # 5 days of fibre outage: 5 x 1,17 €
    ("technical", 83, 0),
])
def test_compensation_is_an_amount_the_grid_can_give(category, cents, expected):
    assert rules.tidy_compensation(category, cents) == expected
