"""Checks the reviewer (callverse/agents/quality).

No Ollama and no PostgreSQL: a fake judge gives fixed grades and the account is made up.
Several calls below are real lines from test runs with llama3.1:8b.
"""

import pytest

from callverse.agents.quality import agent, checks
from callverse.state import QUALITY_CRITERIA, CallState, ComplaintResult, Message, ToolCall
from fakes import FAKE_REVIEW_FACTS, FixedJudge

SEARCHED = [ToolCall(name="get_customer"), ToolCall(name="search_policies", arguments={"question": "x"})]


def finished(*lines, persona="calm", outcome="resolved", satisfaction=4, tools=SEARCHED, **fields) -> CallState:
    roles = ["client", "advisor"] * len(lines)
    return CallState(call_id="q", customer_id="0001-AAAAA", persona=persona, call_reason="check_invoice",
                     messages=[Message(role=r, content=t) for r, t in zip(roles, lines)],
                     turn=(len(lines) + 1) // 2, tools_used=tools, outcome=outcome, satisfaction=satisfaction,
                     **fields)


def report(state: CallState, judge=None, facts=None):
    """The report, with the fake judge's grades (4 everywhere unless told otherwise)."""
    model = judge or FixedJudge()
    judged = None if model.fail else model.with_structured_output(agent.Judgement).invoke("call")
    return agent.review(state, facts or dict(FAKE_REVIEW_FACTS), judged)


GOOD_CALL = finished(
    "Bonjour, je voudrais comprendre ma facture de 49,98 €.",
    "Votre facture de 49,98 € comprend la Box Fibre (34,99 €) et le Forfait Mobile 100 Go (14,99 €). "
    "Puis-je vous aider pour autre chose ?",
    "Parfait, merci, au revoir.")


# ---------------------------------------------------------------- the report

def test_good_call_gets_every_criterion_with_an_explanation():
    r = report(GOOD_CALL)
    assert set(r.scores) == set(QUALITY_CRITERIA)
    assert all(s.explanation for s in r.scores.values())
    assert r.scores["accuracy"].score == 4 and r.scores["compliance"].score == 5
    assert r.scores["resolution"].score == 5 and r.scores["satisfaction"].score == 4
    assert r.overall == pytest.approx(sum(s.score for s in r.scores.values()) / 6, abs=0.01)
    assert r.escalation_correct is None                       # not a complaint


def test_known_amounts_are_not_flagged():
    assert checks.amount_findings(GOOD_CALL, FAKE_REVIEW_FACTS) == []


def test_an_invented_amount_lowers_accuracy_with_the_evidence():
    call = finished("Combien coûte la fibre ?", "La Box Fibre coûte 29,99 € par mois. Puis-je vous aider ?", "Merci.")
    r = report(call)
    assert r.scores["accuracy"].score == 3                    # min(model 4, code 5 - 2)
    assert "29,99 €" in r.scores["accuracy"].explanation and "La Box Fibre coûte 29,99 €" in r.scores["accuracy"].explanation
    assert "Ne donner que des montants trouvés" in r.recommendations[0]


def test_a_changed_compensation_is_a_serious_error():
    # a real answer: the specialist decided 11,70 € and the advisor promised 20 €
    decided = ComplaintResult(category="technical", severity=4, actions=[], compensation_cents=1170)
    call = finished("Mon internet est coupé depuis 4 jours.",
                    "Cependant, le spécialiste a décidé de vous offrir une compensation de 20 €, qui est le plafond "
                    "automatique pour une gravité de 4. Puis-je vous aider pour autre chose ?",
                    "Ce n'est pas assez.", outcome="refused", satisfaction=1, is_complaint=True,
                    complaint_result=decided)
    r = report(call)
    assert r.scores["accuracy"].score == 2                    # 5 - 3
    assert "20,00 €" in r.scores["accuracy"].explanation and "11,70 €" in r.scores["accuracy"].explanation


def test_outage_credits_and_the_bill_plus_an_option_are_known_amounts():
    call = finished("Et si j'ajoute la TV ?", "Votre facture passerait à 57,97 € par mois. Pour 4 jours de coupure, "
                    "l'avoir serait de 3,32 €.", "Merci.")
    assert checks.amount_findings(call, FAKE_REVIEW_FACTS) == []


@pytest.mark.parametrize("line, words", [
    ("Comme votre risque de départ est élevé, je vous transfère.", "risque de départ"),
    ("Pouvez-vous me donner votre mot de passe ?", "mot de passe"),
])
def test_forbidden_words_break_compliance(line, words):
    r = report(finished("Bonjour.", line, "Au revoir."))
    assert r.scores["compliance"].score == 1
    assert words in r.scores["compliance"].explanation


def test_answering_without_looking_anything_up_breaks_compliance():
    r = report(finished("Combien coûte la fibre ?", "34,99 €.", "Merci.", tools=[]))
    assert r.scores["compliance"].score == 3 and "Aucune recherche" in r.scores["compliance"].explanation


def test_an_escalation_must_be_announced():
    escalated = ComplaintResult(category="billing", severity=2, actions=[], escalate=True,
                                escalation_reasons=["menace juridique"])
    silent = finished("Je vais porter plainte !", "Je comprends votre colère.", "Bon.", outcome="escalated",
                      is_complaint=True, complaint_result=escalated)
    told = finished("Je vais porter plainte !", "Un responsable vous recontactera sous 48 heures ouvrées.", "Bon.",
                    outcome="escalated", is_complaint=True, complaint_result=escalated)
    assert report(silent).scores["compliance"].score == 2
    assert report(told).scores["compliance"].score == 5


def test_escalation_choice_is_checked_against_the_rules():
    legal = "Si ce n'est pas réglé, je vais porter plainte et prendre un avocat."
    escalated = ComplaintResult(category="billing", severity=2, actions=[], escalate=True)
    handled = ComplaintResult(category="billing", severity=2, actions=[], escalate=False)

    right = finished(legal, "Un responsable vous recontactera.", "Bon.", is_complaint=True, complaint_result=escalated)
    wrong = finished(legal, "Voici la solution.", "Bon.", is_complaint=True, complaint_result=handled)
    assert report(right).escalation_correct is True
    assert report(wrong).escalation_correct is False           # a legal threat must go to a supervisor

    risky = {**FAKE_REVIEW_FACTS, "churn_risk": 0.8}
    calm = finished("Ma facture est fausse.", "Voici la solution.", "Merci.", is_complaint=True, complaint_result=handled)
    assert report(calm, facts=risky).escalation_correct is False   # churn risk 0.8 >= 0.70


@pytest.mark.parametrize("outcome, turns, score", [("resolved", 2, 5), ("resolved", 7, 4), ("escalated", 2, 4),
                                                   ("refused", 2, 2), ("abandoned", 2, 1)])
def test_resolution_comes_from_the_outcome(outcome, turns, score):
    call = finished("Bonjour.", "Réponse.", "Fin.", outcome=outcome).model_copy(update={"turn": turns})
    assert checks.resolution_score(call)[0] == score


def test_model_failure_gives_neutral_grades_not_a_crash(monkeypatch):
    monkeypatch.setattr(agent, "get_model", lambda: FixedJudge(fail=True))
    r = agent.quality_node(GOOD_CALL)["quality_report"]
    assert r.scores["relevance"].score == 3 and "indisponible" in r.scores["relevance"].explanation
    assert r.scores["compliance"].score == 5                   # the code checks still run


def test_the_judge_sees_the_call_the_tone_rule_and_the_decision(monkeypatch):
    judge = FixedJudge()
    monkeypatch.setattr(agent, "get_model", lambda: judge)
    decided = ComplaintResult(category="billing", severity=2, actions=["Corriger la facture"], compensation_cents=500)
    agent.quality_node(finished("Ma facture a doublé !", "Je comprends.", "Merci.", persona="dissatisfied",
                                is_complaint=True, complaint_result=decided))
    prompt = judge.sent[0]
    assert "Client : Ma facture a doublé !" in prompt and "Conseiller : Je comprends." in prompt
    assert "s'excuser pour la gêne" in prompt                 # the guideline for a dissatisfied customer
    assert "Corriger la facture" in prompt and "5,00 €" in prompt
    assert "Box ADSL : 24,99 €/mois" in prompt                 # the catalogue, to judge accuracy


def test_recommendations_are_short_and_not_repeated():
    tips = report(finished("Combien ?", "29,99 € puis 19,99 € puis 9,49 €.", "Merci.", tools=[])).recommendations
    assert len(tips) <= agent.MAX_RECOMMENDATIONS and len(tips) == len(set(tips))


# ---------------------------------------------------------------- the judge must give evidence

MOBILE_CALL = finished(   # a real call: the advisor was right, the judge said it was wrong
    "Et si je veux conserver mon forfait mobile, dois-je le modifier pour la box ADSL ?",
    "Vous n'aurez pas besoin de modifier ou de renouveler votre forfait mobile de 100 Go pour que cela fonctionne "
    "avec la box ADSL. Les forfaits mobiles ne sont pas liés aux box internet. Puis-je vous aider pour autre chose ?",
    "Merci, au revoir.")


def judge_says(accuracy: dict) -> FixedJudge:
    grade = {"score": 4, "explanation": "Bien."}
    return FixedJudge({"relevance": grade, "empathy": grade, "accuracy": accuracy})


def test_a_criticism_without_a_source_is_not_kept():
    invented = {"score": 3, "explanation": "En réalité, le forfait mobile est lié à la box internet.",
                "quote": "Les forfaits mobiles ne sont pas liés aux box internet.", "contradicts": "aucune"}
    r = report(MOBILE_CALL, judge=judge_says(invented))
    assert r.scores["accuracy"].score == 4
    assert "non retenue" in r.scores["accuracy"].explanation


def test_a_criticism_quoting_words_the_advisor_never_said_is_not_kept():
    fake_quote = {"score": 2, "explanation": "Prix faux.", "quote": "La Box ADSL coûte 19,99 € par mois.",
                  "contradicts": "catalogue"}
    r = report(MOBILE_CALL, judge=judge_says(fake_quote))
    assert r.scores["accuracy"].score == 4 and "ne figure pas dans l'appel" in r.scores["accuracy"].explanation


def test_a_criticism_with_evidence_is_kept():
    real = {"score": 2, "explanation": "Contredit le catalogue.",
            "quote": "Les forfaits mobiles ne sont pas liés aux box internet", "contradicts": "catalogue"}
    assert report(MOBILE_CALL, judge=judge_says(real)).scores["accuracy"].score == 2


def test_explanations_say_who_found_what():
    # a real call: the advisor refunded 38,97 € instead of 37,97 €; the model saw nothing, the code did
    call = finished("Je suis facturé 75,94 € au lieu de 37,97 €.",
                    "Le spécialiste a décidé de vous rembourser le montant facturé à tort de 38,97 €.", "Merci.")
    text = report(call).scores["accuracy"].explanation
    assert text.startswith("Avis du modèle : ")
    assert "Vérification automatique : Montant de 38,97 €" in text
    assert report(GOOD_CALL).scores["compliance"].explanation.startswith("Vérification automatique : ")


def test_a_decided_compensation_must_be_announced():
    decided = ComplaintResult(category="billing", severity=2, actions=[], compensation_cents=500)
    silent = finished("Ma facture est trop chère !", "Je peux vous rappeler.", "D'accord.", is_complaint=True,
                      complaint_result=decided)
    told = finished("Ma facture est trop chère !", "Vous recevrez une compensation de 5,00 €.", "Merci.",
                    is_complaint=True, complaint_result=decided)
    assert report(silent).scores["compliance"].score == 3
    assert "5,00 €" in report(silent).scores["compliance"].explanation
    assert report(told).scores["compliance"].score == 5


def test_no_specialist_decision_to_contradict_in_an_everyday_call():
    invalid = {"score": 3, "explanation": "Contredit la décision.", "contradicts": "décision",
               "quote": "Les forfaits mobiles ne sont pas liés aux box internet"}
    r = report(MOBILE_CALL, judge=judge_says(invalid))
    assert r.scores["accuracy"].score == 4 and "pas de décision du spécialiste" in r.scores["accuracy"].explanation
