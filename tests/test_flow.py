"""Checks that one full call still works from start to finish.

These tests use no AI model and no database, so they run in about a second.
Run them before every pull request:  pytest
"""

import pytest
from pydantic import ValidationError

from callverse.agents.advisor import agent as advisor
from callverse.agents.client import agent as client
from callverse.graph import run_call
from callverse.state import QUALITY_CRITERIA, CallState, ComplaintResult, Message
from fakes import KeywordModel


@pytest.fixture(autouse=True)
def no_ai_model(monkeypatch):
    """The advisor gets a fake model and reads no database or index: no Ollama, no PostgreSQL."""
    monkeypatch.setattr(advisor, "get_model", lambda: KeywordModel(["Votre facture comprend votre forfait."]))
    monkeypatch.setattr(advisor, "prefetch", lambda state: ("", []))


def new_call(call_reason: str) -> CallState:
    return CallState(call_id="test-1", customer_id="7590-VHVEG", persona="calm", call_reason=call_reason)


def test_everyday_call_goes_from_start_to_reviewer():
    done = run_call(new_call("check_invoice"))

    assert done.outcome == "resolved"
    assert done.satisfaction is not None
    assert [m.role for m in done.messages] == ["client", "advisor", "client"]
    assert done.complaint_result is None
    assert set(done.quality_report.scores) == set(QUALITY_CRITERIA)


def test_complaint_call_goes_through_the_specialist():
    done = run_call(new_call("complaint_billing"))

    assert done.is_complaint
    assert isinstance(done.complaint_result, ComplaintResult)
    assert done.outcome is not None
    assert done.quality_report.escalation_correct is not None


def test_call_stops_at_the_turn_limit(monkeypatch):
    def endless_client(state: CallState) -> dict:
        return {"messages": [Message(role="client", content="Et encore une question...")], "turn": state.turn + 1}

    monkeypatch.setattr(client, "client_node", endless_client)
    done = run_call(new_call("check_invoice"))

    assert done.outcome == "abandoned"
    assert done.quality_report is not None


def test_complaint_result_rejects_bad_values():
    with pytest.raises(ValidationError):
        ComplaintResult(category="billing", severity=5, actions=[])        # severity is 1 to 4
    with pytest.raises(ValidationError):
        ComplaintResult(category="weather", severity=2, actions=[])        # unknown category
    with pytest.raises(ValidationError):
        ComplaintResult(category="billing", severity=2, actions=[], compensation_cents=-100)
