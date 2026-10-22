"""Complaint specialist: the steps it follows, in order.

classify -> history -> search rules -> draft fix -> check rules -> decide (back to advisor or to the boss)

STUB: a fixed answer so the whole call can run without an AI model.
The owner replaces complaint_node with the real steps, keeping the same input and output.

Input: the CallState. Output: {"complaint_result": ComplaintResult(...)}
"""

from callverse.state import CallState, ComplaintResult


def complaint_node(state: CallState) -> dict:
    result = ComplaintResult(
        category="billing",
        severity=2,
        actions=["vérifier la facture", "corriger le montant sur la prochaine facture"],
        compensation_cents=0,
        rules_cited=[],
        escalate=False,
    )
    return {"complaint_result": result}
