"""Reviewer: grades each call after it ends and gives tips.

STUB: fixed grades so the whole call can run without an AI model.
The owner replaces quality_node with the real worker, keeping the same input and output.

Input: the finished CallState. Output: {"quality_report": QualityReport(...)}
"""

from callverse.state import QUALITY_CRITERIA, CallState, CriterionScore, QualityReport


def quality_node(state: CallState) -> dict:
    scores = {
        name: CriterionScore(score=3, explanation="Stub grade, not a real review.")
        for name in QUALITY_CRITERIA
    }
    report = QualityReport(
        scores=scores,
        overall=3.0,
        recommendations=["Stub: the real reviewer will write tips here."],
        escalation_correct=None if state.complaint_result is None else True,
    )
    return {"quality_report": report}
