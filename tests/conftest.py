"""Shared test setup.

The complaint specialist and the advisor normally use the AI model, the database and the
rules search. Here they are swapped for offline fakes, so `pytest` still needs no Ollama and no PostgreSQL,
and never writes tickets into your real database.
"""

import pytest

from callverse.agents.advisor import agent as advisor
from callverse.agents.complaint import steps
from fakes import KeywordModel


def offline_tools() -> steps.Tools:
    return steps.Tools(
        llm=None,
        get_history=lambda customer_id, category: steps.History(),
        search=lambda question: [],
        open_ticket=lambda state, result, history: "TCK-TEST",
        update_ticket=lambda ticket_id, new_state: None,
    )


@pytest.fixture(autouse=True)
def complaint_specialist_offline(monkeypatch):
    monkeypatch.setattr(steps, "make_tools", offline_tools)


@pytest.fixture(autouse=True)
def advisor_offline(monkeypatch):
    """A fake advisor model that reads no database and no index (test_advisor.py tests the real reads)."""
    monkeypatch.setattr(advisor, "get_model", lambda: KeywordModel(["Votre facture comprend votre forfait."]))
    monkeypatch.setattr(advisor, "prefetch", lambda state: ("", []))
