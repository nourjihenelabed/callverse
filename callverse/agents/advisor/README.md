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
   still search again, up to 4 times; a search of fewer than 3 words is replaced by the customer's words. When the specialist has decided, its decision (actions,
   compensation, escalation) is given to the model, which explains exactly that to the customer.

## Tools

| Tool | What it gives | Where it reads |
|---|---|---|
| `search_policies(question)` | The rule sections that answer the question (prices, delays, procedures) | Chroma index of `data/policies/` |
| `get_complaint_history()` | The caller's 5 latest complaints and their status | `complaints` |
| customer file (always read first, not a model tool) | The caller's offers, contract, monthly bill, payment method | `customers`, `customer_products` |

The customer data is tied to the caller: the model cannot read another customer's file.
They never read `churned` (the real answer, kept for evaluation) nor `churn_risk` (internal).

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

- `pytest tests/test_advisor.py`: 18 tests with a scripted fake model, a temporary SQLite database
  and a fake search model. No Ollama needed.
- `python scripts/try_advisor.py`: the real model, database and rules search.
