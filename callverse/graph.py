"""Decides which worker speaks next during a call (built with LangGraph).

Shared by everyone. Ask the team before changing it.

The order of a call:

    client -> advisor -> client -> advisor -> ... -> reviewer
                 |  ^
                 v  |
            complaint specialist   (only when the advisor sets is_complaint)

- The call ends when the client sets an outcome, or when it reaches
  call.max_turns in settings.yaml (then the outcome is "abandoned").
- The reviewer always runs last.

Usage:
    from callverse.graph import run_call
    finished = run_call(CallState(call_id="c1", customer_id="7590-VHVEG",
                                  persona="calm", call_reason="check_invoice"))
"""

from langgraph.graph import END, START, StateGraph

from callverse.agents.advisor import agent as advisor
from callverse.agents.client import agent as client
from callverse.agents.complaint import steps as complaint
from callverse.agents.quality import agent as quality
from callverse.config import load_settings
from callverse.state import CallState


def _max_turns() -> int:
    return load_settings()["call"]["max_turns"]


def end_call(state: CallState) -> dict:
    """Closes a call that hit the turn limit without an outcome."""
    if state.outcome is None:
        return {"outcome": "abandoned"}
    return {}


def after_client(state: CallState) -> str:
    return "quality" if state.outcome is not None else "advisor"


def after_advisor(state: CallState) -> str:
    if state.is_complaint and state.complaint_result is None:
        return "complaint"
    if state.turn >= _max_turns():
        return "end_call"
    return "client"


def build_graph():
    """Builds the call flow. Worker functions are read when this runs, so tests can swap them."""
    graph = StateGraph(CallState)

    graph.add_node("client", client.client_node)
    graph.add_node("advisor", advisor.advisor_node)
    graph.add_node("complaint", complaint.complaint_node)
    graph.add_node("end_call", end_call)
    graph.add_node("quality", quality.quality_node)

    graph.add_edge(START, "client")
    graph.add_conditional_edges("client", after_client, ["advisor", "quality"])
    graph.add_conditional_edges("advisor", after_advisor, ["complaint", "client", "end_call"])
    graph.add_edge("complaint", "advisor")
    graph.add_edge("end_call", "quality")
    graph.add_edge("quality", END)

    return graph.compile()


def run_call(state: CallState) -> CallState:
    """Runs one whole call and returns the finished case file."""
    result = build_graph().invoke(state, {"recursion_limit": 4 * _max_turns() + 10})
    return CallState.model_validate(result)
