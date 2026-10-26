# Advisor

The only worker the customer talks to. It runs with the AI model (`llama3.1:8b` in Ollama).

## One advisor turn

1. **Understand.** The model reads the customer's last message and decides the `intent`
   (for example `lost_sim`, `check_invoice`) and whether it is a complaint.
2. **Complaint?** If yes and the complaint specialist has not decided yet, the advisor returns
   `{"is_complaint": True}` with no message. The graph sends the call to the specialist, then back here.
3. **Answer.** First the code itself reads the customer's file and searches the rules with the
   customer's own words, and puts both in the model's instructions. A small model like llama3.1:8b
   often forgets a tool or writes a bad search (it once searched for just "J" from "J'ai perdu..."),
   so it starts with what it needs. Then the model answers in French, in at most 4 sentences. It may
   still search again, up to 4 times; a search of fewer than 3 words is replaced by the customer's words.
   Two more checks are done by code, because the small model skips them:
   - if the customer talks about a SIM, a mobile line or a box they don't have, an explicit warning
     is added ("le client n'a pas de ligne mobile") so it doesn't recite the procedure;
   - if the specialist sent the case to a supervisor, the customer always hears that a manager will call
     back within 48 working hours (added by code if the model forgets). The model is told the specialist's
     actions are only ideas for the manager, so it promises no refund, and it never sees the reason
     (it can be the churn risk, which is internal);
   - the specialist's compensation cannot be changed: a sentence about compensation with another amount
     is replaced by the decided amount (the model once promised 20 € instead of the decided amount);
   - answers are cut after 5 sentences, and only one question is asked at the end;
   - placeholders ("[Nom]") and stage directions ("(pause)") are removed;
   - an answer 80% identical to one already given in the call is asked again once; if it repeats again,
     the advisor says in one sentence that it has no more information (the model once looped 4 times);
   - if the model writes its tool call as text (`{"name": "search_policies", ...}`), the tool is run
     and the customer never hears the JSON. When the specialist has decided, its decision (actions,
   compensation, escalation) is given to the model, which explains exactly that to the customer.

## Tools

| Tool | What it gives | Where it reads |
|---|---|---|
| `search_policies(question)` | The rule sections that answer the question (prices, delays, procedures) | Chroma index of `data/policies/` |
| `get_complaint_history()` | The caller's 5 latest complaints and their status | `complaints` |
| customer file (always read first, not a model tool) | The caller's offers, contract, monthly bill, payment method | `customers`, `customer_products` |

The customer data is tied to the caller: the model cannot read another customer's file.
They never read `churned` (the real answer, kept for evaluation) nor `churn_risk` (internal).

The whole catalogue (10 offers and prices) is always in the instructions: the model once said the
Box ADSL did not exist. It may not invent discounts or "special offers": only a manager can.

## Rules it follows (from `data/policies/agent_guidelines.md`)

- Search the rules before giving a price, delay or rule; never answer from memory.
- If the information is not found, say so and offer a call back.
- Never promise a compensation: only the complaint specialist decides.
- Never ask for a password or a full card number.

## Files

| File | What it does |
|---|---|
| `agent.py` | `advisor_node`: the three steps above |
| `tools.py` | The three tools |
| `prompts.py` | The instructions given to the model, in French |

## Checks

- `pytest tests/test_advisor.py`: 33 tests with a scripted fake model, a temporary SQLite database
  and a fake search model. No Ollama needed.
- `python scripts/try_advisor.py`: the real model, database and rules search.
