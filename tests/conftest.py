"""Shared test setup.

The complaint specialist normally uses the AI model, the database and the rules search.
Here they are swapped for offline fakes, so `pytest` still needs no Ollama and no PostgreSQL,
and never writes tickets into your real database.
"""

import pytest

from callverse.agents.complaint import steps


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
