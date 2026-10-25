"""Checks the advisor (callverse/agents/advisor).

No Ollama and no PostgreSQL needed: a scripted fake model replaces the AI model,
a temporary SQLite file replaces the database, and a word-counting embedding
replaces the search model.
"""

import json

import pytest
from langchain_core.messages import SystemMessage, ToolMessage
from sqlalchemy import create_engine

from callverse import knowledge
from callverse.agents.advisor import agent, prompts, tools
from callverse.config import load_settings
from callverse.state import CallState, ComplaintResult, Message
from fakes import ScriptedModel, tool_call
from test_knowledge import WordCountEmbeddings
from test_seed import fake_rows, seed

REAL_PREFETCH = agent.prefetch   # conftest.py replaces it for the other tests; here we test the real one
CUSTOMER = "0001-AAAAA"          # fibre, mobile, extra line, security, TV; a churner in the IBM data


@pytest.fixture(autouse=True)
def no_real_database(monkeypatch):
    """Tests never touch your PostgreSQL. The `database` fixture gives a SQLite one instead."""
    def refuse():
        raise RuntimeError("no database in this test")
    monkeypatch.setattr(tools, "get_engine", refuse)
    monkeypatch.setattr(agent, "prefetch", REAL_PREFETCH)


@pytest.fixture
def database(tmp_path, monkeypatch):
    rows, products = seed.build_customers(fake_rows())
    complaint_rows, ticket_rows = seed.build_complaints(rows, seed=1)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    seed.write_all(engine, rows, products, complaint_rows, ticket_rows)
    monkeypatch.setattr(tools, "get_engine", lambda: engine)
    return complaint_rows


@pytest.fixture
def rules(tmp_path, monkeypatch):
    monkeypatch.setitem(load_settings()["rag"], "store", "local")
    fake = WordCountEmbeddings()
    knowledge.build_index(embeddings=fake, index_dir=tmp_path / "index")
    real_search = knowledge.search_policies
    monkeypatch.setattr(knowledge, "search_policies",
                        lambda q: real_search(q, k=2, embeddings=fake, index_dir=tmp_path / "index"))


def use(monkeypatch, model):
    monkeypatch.setattr(agent, "get_model", lambda: model)
    return model


def call(*client_messages, **fields) -> CallState:
    return CallState(call_id="t", customer_id=CUSTOMER, persona="calm", call_reason="check_invoice",
                     messages=[Message(role="client", content=m) for m in client_messages], **fields)


# ---------------------------------------------------------------- the tools

def test_customer_file_has_the_offers_and_no_churn_fields(database):
    file = json.loads(tools.customer_file(CUSTOMER))
    assert file["facture_mensuelle"] == "71,95 €"           # 34,99 + 14,99 + 9,99 + 3,99 + 7,99
    assert "Box Fibre (34,99 €/mois)" in file["offres"]
    assert file["contrat"] == "sans engagement"
    text = json.dumps(file).lower()
    assert "churn" not in text and "risk" not in text and "risque" not in text


def test_unknown_customer_is_said_plainly(database):
    assert "Aucun client" in tools.customer_file("9999-ZZZZZ")


def test_complaint_history_lists_past_complaints(database):
    mine = [c for c in database if c["customer_id"] == CUSTOMER]
    result = tools.complaint_history(CUSTOMER)
    if mine:
        assert len(json.loads(result)) == min(len(mine), tools.MAX_COMPLAINTS)
    else:
        assert result.startswith("Aucune")


def test_catalogue_names_match_the_seed_prices():
    assert {code: cents for code, (_, cents) in tools.catalogue().items()} == seed.CATALOGUE


def test_policy_search_returns_rule_sections(rules):
    text = tools.policy_search("Combien coûte une nouvelle carte SIM après une perte ?")
    assert text.startswith("[") and ".md >" in text


def test_complaint_history_tool_only_sees_the_caller(database):
    history = {t.name: t for t in tools.build_tools("0002-BBBBB")}["get_complaint_history"]
    asked_for_someone_else = history.invoke({"customer_id": CUSTOMER})
    assert asked_for_someone_else == tools.complaint_history("0002-BBBBB")


# ---------------------------------------------------------------- the advisor

def test_complaint_goes_to_the_specialist_first(monkeypatch):
    model = use(monkeypatch, ScriptedModel(intent="complaint_billing", is_complaint=True))
    update = agent.advisor_node(call("Ma facture a doublé, c'est inadmissible !"))
    assert update == {"intent": "complaint_billing", "is_complaint": True}     # no message yet
    assert "inadmissible" in model.sent[0]


def test_customer_file_and_rules_are_read_before_the_model_answers(monkeypatch, rules, database):
    model = use(monkeypatch, ScriptedModel(["Vous payez 71,95 € par mois. Puis-je vous aider pour autre chose ?"]))
    update = agent.advisor_node(call("J'ai perdu ma carte SIM, ça coûte combien ?"))

    system = model.sent[-1][0].content
    assert "71,95 €" in system and "Box Fibre" in system                # the customer's file
    assert ".md >" in system                                            # the rule sections found
    assert [(c.name, c.arguments) for c in update["tools_used"]] == [
        ("get_customer", {}), ("search_policies", {"question": "J'ai perdu ma carte SIM, ça coûte combien ?"})]


def test_model_can_search_again(monkeypatch, rules, database):
    model = use(monkeypatch, ScriptedModel([
        tool_call("search_policies", question="prix nouvelle carte SIM"),
        "Une nouvelle carte SIM coûte 10 €. Puis-je vous aider pour autre chose ?",
    ], intent="lost_sim"))
    update = agent.advisor_node(call("J'ai perdu ma carte SIM, ça coûte combien ?"))

    assert update["intent"] == "lost_sim" and update["is_complaint"] is False
    assert update["messages"][0].role == "advisor"
    assert "carte SIM" in update["messages"][0].content
    assert update["tools_used"][-1].arguments == {"question": "prix nouvelle carte SIM"}
    assert set(model.tool_names) == {"search_policies", "get_complaint_history"}
    second = model.sent[-1]                                  # the model saw the search result
    assert isinstance(second[-1], ToolMessage) and ".md >" in second[-1].content


def test_search_cut_short_uses_the_customer_words(monkeypatch, rules):
    use(monkeypatch, ScriptedModel([tool_call("search_policies", question="J"), "D'accord."]))
    update = agent.advisor_node(call("J'ai perdu ma carte SIM, que dois-je faire ?"))
    assert update["tools_used"][-1].arguments == {"question": "J'ai perdu ma carte SIM, que dois-je faire ?"}


def test_no_database_and_no_index_still_gives_an_answer(monkeypatch):
    monkeypatch.setattr(knowledge, "search_policies", lambda q: (_ for _ in ()).throw(RuntimeError("no index")))
    model = use(monkeypatch, ScriptedModel(["Je vais vérifier et vous rappeler."]))
    update = agent.advisor_node(call("Bonjour"))
    system = model.sent[-1][0].content
    assert "Dossier indisponible" in system and "Recherche indisponible" in system
    assert update["messages"][0].content == "Je vais vérifier et vous rappeler."


def test_system_prompt_never_shows_the_churn_risk(monkeypatch, database):
    model = use(monkeypatch, ScriptedModel(["Votre facture est de 71,95 €."]))
    agent.advisor_node(call("Combien je paie par mois ?"))
    everything = " ".join(str(m.content) for m in model.sent[-1]).lower()
    assert "71,95" in everything
    assert "churn" not in everything and "risque de départ" not in everything


def test_intent_is_kept_on_later_turns(monkeypatch):
    use(monkeypatch, ScriptedModel(["D'accord."], intent="something_else"))
    update = agent.advisor_node(call("Merci, et pour le mobile ?", intent="check_invoice"))
    assert "intent" not in update


def test_too_many_tool_calls_forces_an_answer(monkeypatch, rules):
    loops = [tool_call("search_policies", call_id=f"c{i}", question="encore") for i in range(agent.MAX_TOOL_ROUNDS)]
    use(monkeypatch, ScriptedModel(loops + ["Voici ce que j'ai trouvé."]))
    update = agent.advisor_node(call("Une question compliquée"))
    assert len(update["tools_used"]) == 2 + agent.MAX_TOOL_ROUNDS          # 2 read before + the model's searches
    assert update["messages"][0].content == "Voici ce que j'ai trouvé."


def test_broken_tool_does_not_stop_the_call(monkeypatch):
    def broken(question):
        raise RuntimeError("index not built")

    monkeypatch.setattr(knowledge, "search_policies", broken)
    model = use(monkeypatch, ScriptedModel([tool_call("search_policies", question="x"), tool_call("fly_away"),
                                            "Je vais vérifier et vous rappeler."]))
    update = agent.advisor_node(call("Bonjour"))
    tool_messages = [m for m in model.sent[-1] if isinstance(m, ToolMessage)]
    assert "Erreur" in tool_messages[0].content and "index not built" in tool_messages[0].content
    assert "Recherche indisponible" in model.sent[-1][0].content
    assert "n'existe pas" in tool_messages[1].content
    assert update["messages"][0].content == "Je vais vérifier et vous rappeler."


def test_empty_answer_gets_a_polite_fallback(monkeypatch):
    use(monkeypatch, ScriptedModel(["   "]))
    assert agent.advisor_node(call("Bonjour"))["messages"][0].content == agent.FALLBACK_ANSWER


def decision(escalate: bool) -> ComplaintResult:
    return ComplaintResult(
        category="billing", severity=3, compensation_cents=1500, rules_cited=["compensation_grid.md"],
        actions=["Corriger la facture"] + (["informer le client que son dossier est transmis à un responsable, "
                                            "qui le recontacte sous 48 heures ouvrées"] if escalate else []),
        escalate=escalate, escalation_reasons=["risque élevé de résiliation"] if escalate else [])


def complaint_call(escalate: bool) -> CallState:
    return call("Ma facture a doublé !", intent="complaint_billing", is_complaint=True, complaint_result=decision(escalate))


def test_specialist_decision_is_given_to_the_model(monkeypatch):
    model = use(monkeypatch, ScriptedModel(["Nous corrigeons votre facture et vous remboursons 15 €."]))
    update = agent.advisor_node(complaint_call(escalate=False))

    system = model.sent[0][0]
    assert isinstance(system, SystemMessage)
    assert "Corriger la facture" in system.content and "15,00 €" in system.content
    assert system.content.rstrip().endswith(prompts.COMPLAINT_REMINDER.strip()[-60:])   # the last words it reads
    assert "is_complaint" not in update                      # no new routing: the call goes back to the customer
    assert update["messages"][0].content.startswith("Nous corrigeons")


def test_escalated_complaint_promises_nothing_and_hides_the_reason(monkeypatch):
    model = use(monkeypatch, ScriptedModel(["Je comprends."]))
    agent.advisor_node(complaint_call(escalate=True))
    system = model.sent[0][0].content
    assert "résiliation" not in system                       # the churn risk is internal
    assert "pistes" in system and "AUCUN remboursement" in system


def test_escalation_is_said_even_if_the_model_forgets(monkeypatch):
    use(monkeypatch, ScriptedModel(["Je comprends votre colère. Puis-je vous aider pour autre chose ?"]))
    text = agent.advisor_node(complaint_call(escalate=True))["messages"][0].content
    assert text == ("Je comprends votre colère. " + prompts.ESCALATION_SENTENCE + " " + prompts.CLOSING)

    use(monkeypatch, ScriptedModel(["Un responsable vous rappellera sous 48 heures."]))
    text = agent.advisor_node(complaint_call(escalate=True))["messages"][0].content
    assert text == "Un responsable vous rappellera sous 48 heures."            # already said: unchanged


def test_tool_call_with_a_bad_escape_is_still_run():
    as_text = '{"name": "search_policies", "parameters": {"question": "quelles offres j\\\'ai ?"}}'
    reply = agent.text_tool_call(agent.AIMessage(as_text), {"search_policies": None})
    assert reply.tool_calls[0]["args"] == {"question": "quelles offres j'ai ?"}


def test_text_json_fallback_for_understanding():
    state = call("Bonjour")
    assert agent.parse_understanding('Voici : {"intent": "lost_sim", "is_complaint": false}', state).intent == "lost_sim"
    assert agent.parse_understanding("je ne sais pas", state).is_complaint is False


def test_customer_without_a_mobile_line_gets_a_warning(monkeypatch, database):
    model = use(monkeypatch, ScriptedModel(["Votre compte ne comporte pas de ligne mobile."]))
    state = call("J'ai perdu ma carte SIM, qu'est-ce que je dois faire ?").model_copy(update={"customer_id": "0002-BBBBB"})
    agent.advisor_node(state)                                # 0002-BBBBB has only a Box ADSL
    assert "n'a pas de ligne mobile" in model.sent[-1][0].content


def test_no_warning_when_the_customer_has_the_product_or_asks_a_price(monkeypatch, database):
    model = use(monkeypatch, ScriptedModel(["a", "b"]))
    agent.advisor_node(call("J'ai perdu ma carte SIM"))          # 0001-AAAAA has a mobile line
    agent.advisor_node(call("Combien coûte un forfait mobile ?").model_copy(update={"customer_id": "0002-BBBBB"}))
    assert all("ATTENTION" not in sent[0].content for sent in model.sent if isinstance(sent, list))


def test_tool_call_written_as_text_is_run_not_said(monkeypatch, rules):
    as_text = '{"name": "search_policies", "parameters": {"question": "Combien je paie par mois ?"}}'
    model = use(monkeypatch, ScriptedModel([as_text, "Vous payez 71,95 €."]))
    update = agent.advisor_node(call("Combien je paie par mois ?"))
    assert update["messages"][0].content == "Vous payez 71,95 €."      # the JSON never reaches the customer
    assert update["tools_used"][-1].arguments == {"question": "Combien je paie par mois ?"}
    assert isinstance(model.sent[-1][-1], ToolMessage)


def test_ordinary_text_starting_with_a_brace_is_kept():
    reply = agent.AIMessage('{"note": "pas un outil"}')
    assert agent.text_tool_call(reply, {"search_policies": None}) is reply


def test_an_invented_compensation_is_replaced_by_the_decided_one():
    state = complaint_call(escalate=False)                                   # the specialist decided 15 €
    text = agent.finish("Je comprends. Le spécialiste a décidé de vous offrir une compensation de 20 €. "
                        "Puis-je vous aider pour autre chose ?", state)
    assert "20 €" not in text and "15,00 €" in text


def test_a_refund_amount_is_not_touched():
    state = complaint_call(escalate=False)
    text = "Nous vous remboursons 27,98 € facturés à tort. Puis-je vous aider pour autre chose ?"
    assert agent.finish(text, state) == text


def test_long_answers_are_cut_and_only_one_question_is_asked():
    long = " ".join(f"Phrase {n}." for n in range(1, 9)) + " " + prompts.CLOSING
    assert agent.tidy_answer(long) == " ".join(f"Phrase {n}." for n in range(1, agent.MAX_SENTENCES + 1)) + " " + prompts.CLOSING
    two = f"Je peux le changer. {prompts.ASK_CONSENT} {prompts.CLOSING}"
    assert agent.tidy_answer(two) == f"Je peux le changer. {prompts.ASK_CONSENT}"


def test_the_whole_catalogue_is_always_given():
    prompt = agent.system_prompt(call("Bonjour"))
    for name, cents in tools.catalogue().values():
        assert f"{name} : {tools.euros(cents)}/mois" in prompt
    assert "offre spéciale" in prompt                                  # no invented loyalty deals


def test_placeholders_and_stage_directions_are_removed():
    assert agent.clean("Je m'appelle [Nom], je suis là pour vous aider. (pause) Oui, c'est bien ainsi.") == \
        "Je suis là pour vous aider. Oui, c'est bien ainsi."


def test_no_closing_question_after_another_question():
    assert agent.tidy_answer(f"Quel est le montant exact ? {prompts.CLOSING}") == "Quel est le montant exact ?"
