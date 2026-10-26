# Supervisor (retention route)

Added after the first night batch: the 10 demanding and churn-likely customers were never resolved, and
nobody could make them an offer. The company rules say what to do:

- `agent_guidelines.md`, "Client qui veut partir": listen to the reason, explain the cancellation
  conditions (`contracts.md`), and hand the case to a supervisor if the churn risk is high.
- `compensation_grid.md`, "Offres de fidélisation": only a supervisor may offer, to a customer at high
  risk, up to 3 free months on an option (e.g. TV or Cinéma & Séries), or a free move from the Box ADSL to
  the Box Fibre if the home is eligible.

## How it works

1. **The advisor notices the customer wants to leave**: the AI model says so (`wants_to_leave`), or the code
   finds words like "je veux résilier", "changer d'opérateur", "code RIO". Contesting old cancellation fees
   is not wanting to leave.
2. **The call goes to the supervisor once** (`agent.py`, plain code, no AI model), which reads the churn risk
   (from the churn model) and the customer's products:

   | Churn risk | Products | Decision |
   |---|---|---|
   | below 0.70 | any | no supervisor, no offer: the advisor explains the cancellation conditions |
   | 0.70 or more | Box ADSL | free move to the Box Fibre, same price, if the home is eligible |
   | 0.70 or more | Box Fibre | 3 free months on TV (or Cinéma & Séries if they already have TV) |
   | 0.70 or more | mobile only | no offer in the grid: a supervisor calls back within 48 working hours |

3. **The advisor presents the decision**, as coming from a supervisor. It never sees or says the churn score.
4. **The reviewer checks** that the offer (or the call-back) was really said to the customer.

The threshold (0.70) is the same as the complaint rules' (`complaint/rules.py`).
Tests: `tests/test_retention.py`.
