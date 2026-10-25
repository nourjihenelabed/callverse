# Pretend customer (client simulator)

Plays the person who calls, so whole calls can run on their own. It runs with the AI model
(`llama3.1:8b`, temperature 0.7 so two calls are not identical).

## One call

1. **A new call** (`scenarios.random_call`): a customer from the database, a reason that fits
   their products (no lost SIM for a customer without a mobile line), and a mood.
   - The reasons follow the workforce simulation's queue shares: 55% general, 30% technical, 15% complaints.
   - Customers with a high churn risk often call as `likely_to_churn`. The simulation may read the
     churn risk because it plays the real customer; the advisor never sees it.
2. **First turn:** the model writes the opening sentence from the mood and the situation, filled with
   the customer's real account (for example "ma facture est de 55,96 € au lieu de 27,98 €").
3. **Next turns:** the model reads the advisor's answer and decides `continue`, `accept`, `refuse` or
   `hang_up`, with a satisfaction from 1 to 5.
4. **Rules kept by code:**
   - each mood has a patience limit; when it runs out, the customer hangs up (`abandoned`, satisfaction 2 at most);
   - accepting an escalated complaint ends the call as `escalated`;
   - the decision must match the words: a goodbye counts as `accept`, a new question as `continue`;
   - placeholders like "[votre nom]" are removed;
   - if the model fails, the customer ends the call politely instead of stopping the program.

## The six moods

| Mood | Patience (messages) | How the customer talks |
|---|---|---|
| calm | 8 | polite, clear, accepts a reasonable answer |
| impatient | 4 | short sentences, wants an answer now |
| dissatisfied | 6 | wants the problem acknowledged first |
| demanding | 7 | wants exact amounts and a gesture, insists |
| interested_in_offer | 8 | asks about other offers and prices |
| likely_to_churn | 5 | talks about leaving for a competitor |

## Files

| File | What it does |
|---|---|
| `agent.py` | `client_node`: one customer turn |
| `personas.py` | The six moods and their patience |
| `scenarios.py` | The 16 call reasons, the customer's account facts, `random_call` |
| `prompts.py` | The instructions given to the model, in French |

## Checks

- `pytest tests/test_client.py`: 25 tests with a scripted fake model and a temporary SQLite database.
- `python scripts/try_call.py`: whole calls with the real customer, advisor and complaint specialist.
