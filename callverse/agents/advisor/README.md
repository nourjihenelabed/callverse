# Advisor

The only worker the customer talks to. It runs with the AI model (`llama3.1:8b` in Ollama).

## One advisor turn

1. **Understand.** The model reads the customer's last message and decides the `intent`
   (for example `lost_sim`, `check_invoice`) and whether it is a complaint.
2. **Complaint?** If yes and the complaint specialist has not decided yet, the advisor returns
   `{"is_complaint": True}` with no message. The graph sends the call to the specialist, then back here.
3. **Answer.** The model answers in French, 2 to 4 sentences, using its tools. It may call tools
   up to 4 times, then it must answer. When the specialist has decided, its decision (actions,
   compensation, escalation) is given to the model, which explains exactly that to the customer.

## Tools

| Tool | What it gives | Where it reads |
|---|---|---|
| `search_policies(question)` | The rule sections that answer the question (prices, delays, procedures) | Chroma index of `data/policies/` |
| `get_customer()` | The caller's offers, contract, monthly bill, payment method | `customers`, `customer_products` |
| `get_complaint_history()` | The caller's 5 latest complaints and their status | `complaints` |

The customer tools are tied to the caller: the model cannot read another customer's file.
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

- `pytest tests/test_advisor.py`: 15 tests with a scripted fake model, a temporary SQLite database
  and a fake search model. No Ollama needed.
- `python scripts/try_advisor.py`: the real model, database and rules search.
