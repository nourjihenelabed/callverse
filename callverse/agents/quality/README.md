# Reviewer (quality)

Grades every finished call on 6 criteria, from 1 to 5, and **explains each grade** (the explainability
part of the project). It mixes the AI model (`llama3.1:8b`, temperature 0) with checks in plain code:
the model judges what needs reading, the code checks what can be checked exactly.

| Criterion | Who grades | How |
|---|---|---|
| relevance | AI model | Did the advisor answer what the customer asked? |
| accuracy | AI model + code | The model judges the facts; the code checks every euro amount the advisor said against the rules, the customer's bill and offers, what the customer said and the specialist's decision. An unknown amount costs 2 points, a changed compensation 3. The lower grade wins. |
| compliance | code | The rules of `agent_guidelines.md` that code can check: rules or file searched before answering; churn risk never mentioned; no password or card number asked; complaints sent to the specialist; a decided compensation or an escalation announced to the customer. |
| empathy | AI model | Did the tone fit the customer's mood, as `agent_guidelines.md` ("Adapter son ton au client") says? |
| resolution | code | From the outcome: resolved 5 (4 if more than 5 customer messages), escalated 4, refused 2, abandoned 1. |
| satisfaction | code | The customer's own score. |

- **overall**: the average of the 6 grades.
- **recommendations**: up to 4 tips, first those of the code checks, then the model's.
- **escalation_correct** (complaints only): the 4 supervisor rules (legal threat, 3rd complaint of the same
  category in 12 months, churn risk ≥ 0.70, compensation over the automatic cap) recomputed from the call
  and compared with the specialist's choice.

**The judge must give evidence.** To lower accuracy to 3 or less, the model must copy the advisor's exact
wrong sentence and say which source it contradicts (catalogue, customer file or specialist's decision).
If the sentence is not in the call, or it contradicts none of them, the criticism is not kept and the grade
goes back up to 4, with the reason written in the explanation. (In a test run, the 8B judge invented an
error: it said a mobile plan is tied to the internet box, which no rule says.)

**Every explanation says who found what:** `Avis du modèle : ...` for the model, `Vérification automatique : ...`
for the code. Every finding of the code quotes the advisor's sentence, for example:
`Compensation de 20,00 € annoncée, alors que le spécialiste a décidé 11,70 € : « ... »`.

If the model fails, relevance, accuracy and empathy get a neutral 3 with the reason; the code checks still run.

## Files

| File | What it does |
|---|---|
| `agent.py` | `quality_node`: runs the model and the checks, builds the `QualityReport` |
| `checks.py` | The code checks, each with its evidence and its tip |
| `prompts.py` | The judge's instructions and the tone rule for each mood |

## Checks

- `pytest tests/test_quality.py`: 23 tests, several built from real calls of the test runs.
- `python scripts/try_call.py`: whole calls with the review printed after each one.
