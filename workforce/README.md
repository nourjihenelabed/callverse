# Workforce manager

The floor manager decides, every 15 minutes, how many advisors work on each queue (general, technical, complaint). It is trained with reinforcement learning (PPO) and compared with three classic rules on days it never saw.

## Files

| File | What it does |
|---|---|
| `simulation.py` | The simulated call center (SimPy): one day of calls, three queues, 30 advisors with skills (24 planned + 6 on call), customers who hang up, realistic staffing rules, test scenarios, and the KPIs. |
| `manager.py` | The rules: Erlang C plan, margin rule, reactive rule. |
| `env.py` | The simulation as a Gymnasium environment, random training days, the reward, and `RLPolicy` to use a trained agent like any rule. |
| `train.py` | Trains an agent with PPO (Stable-Baselines3). |
| `compare.py` | Compares all policies on 8 test scenarios, 30 days each, with a statistical test. |
| `explain.py` | Charts that explain the agent's decisions (XAI). |

## How to run

```bash
pip install -e ".[dev,rl]"            # RL packages, including PyTorch (about 1 GB) and Plotly
python -m workforce.simulation        # one day with the Erlang C plan
python -m workforce.train --seed 0    # about 20 minutes on a laptop; repeat with --seed 1 and --seed 2
python -m workforce.compare           # the comparison tables (saved to workforce/models/comparison.json)
python -m workforce.explain           # the XAI charts (workforce/models/explain/*.html)
```

## The simulated call center

- About 1,050 calls a day (8h–20h), more around 10h and 14h: 55% general, 30% technical, 15% complaints.
- 30 advisors: 9 general only, 12 general + technical, 9 general + complaint. Paid 25 €/hour when on duty.
- A customer hangs up if they wait longer than their patience (3 to 5 minutes on average).
- **Realistic staffing rules**, the same for every policy: an advisor called in stays **at least 1 hour**; moving to another queue takes **2 minutes**; a policy can turn on **priority** (multi-skilled advisors on general calls serve waiting technical or complaint calls first).

## The policies

| Policy | What it does |
|---|---|
| **Erlang C plan** | Staffing computed in advance from the forecast with the Erlang C formula (80% of calls answered within 20 s), hour by hour. Never changes during the day. |
| **Reactive rule** | The Erlang C plan, plus one advisor per waiting customer (at most 5 per queue). |
| **Margin rule** | The Erlang C plan, plus 4 general advisors, priority on. Found by analyzing what the first RL agent did (see "How we got here"). |
| **RL agent (PPO)** | Every 15 minutes, corrects the Erlang C plan by −3 to +3 advisors per queue and turns priority on or off. |

The two rules' settings (and whether to use priority) were tuned on random training days, never on the test days.

## The RL problem

- **Episode:** one day; **step:** one decision every 15 minutes (52 per day).
- **What the agent sees (20 numbers):** time of day; per queue: customers waiting, advisors working, average wait in the last 15 minutes, Erlang C plan, and **calls received in the last 15 minutes compared with the forecast**; plus last period's SLA and hang-ups, the forecast for the next hour, and whether priority is on.
- **What it decides:** a correction to the Erlang C plan (−3 to +3 advisors per queue) and priority on/off.
- **Reward:** minus the cost of the 15 minutes in euros: staff pay + **50 € per customer who hangs up** + **8 € per call answered after 20 s**.
- **Training:** 3 agents (seeds 0, 1, 2), 400,000 decisions each (about 7,700 simulated days), on **random days**: 25% normal, 15% quieter than forecast, 60% with 1 or 2 surprise peaks of random queue, time, length and size. The agent is never told what kind of day it is.

## Test scenarios (never used in training)

| Scenario | What happens (the forecast does not know) |
|---|---|
| normal | The usual day |
| network outage | Technical calls ×2 from 10h to 13h, complaints ×1.5 from 11h to 14h |
| billing day | General ×1.3 and complaints ×1.6 all day |
| afternoon promo | General calls ×1.8 from 14h to 17h |
| late outage | The outage, but from 16h |
| big outage | Technical ×2.5 from 10h to 13h, complaints ×1.8 from 11h to 14h |
| complaint storm | Complaints ×2.5 from 9h to 12h |
| quiet day | 30% fewer calls than forecast, all day |

## Results

Same 30 test days per scenario for every policy (days 10000–10029). RL = average of the 3 agents ± the spread between them.

### Total cost per day (staff + price of bad service)

| Scenario | Erlang C plan | Reactive rule | Margin rule | RL agent | Best |
|---|---|---|---|---|---|
| normal | 7,857 € | 7,664 € | **7,531 €** | 7,664 € ±102 | Margin rule |
| network outage | 10,827 € | 9,591 € | **8,883 €** | 9,160 € ±177 | Margin rule |
| billing day | 12,305 € | 10,707 € | **9,325 €** | 9,661 € ±388 | Margin rule |
| afternoon promo | 11,651 € | 10,044 € | **8,480 €** | 9,197 € ±110 | Margin rule |
| late outage | 9,906 € | 9,097 € | 8,992 € | **8,621 € ±149** | RL agent |
| big outage | 13,491 € | 11,189 € | 10,946 € | **10,805 € ±86** | RL agent |
| complaint storm | 10,105 € | **9,154 €** | 9,553 € | 9,238 € ±189 | Reactive rule |
| quiet day | **6,004 €** | 6,015 € | 6,909 € | 6,564 € ±183 | Erlang C plan |
| **average** | 10,268 € | 9,183 € | **8,827 €** | 8,864 € | |
| **worst gap to the best policy** | 3,171 € | 1,564 € | 905 € | **717 €** | |

### Is the RL agent really better? (paired test on the same 30 days)

| Scenario | RL vs margin rule | RL vs reactive rule |
|---|---|---|
| normal | −133 €/day, better on 9/30 (significant) | 0 €/day, better on 15/30 (not significant) |
| network outage | −276 €/day, better on 8/30 (significant) | +431 €/day, better on 23/30 (significant) |
| billing day | −336 €/day, better on 8/30 (significant) | +1,046 €/day, better on 29/30 (significant) |
| afternoon promo | −717 €/day, better on 1/30 (significant) | +847 €/day, better on 28/30 (significant) |
| late outage | +371 €/day, better on 27/30 (significant) | +476 €/day, better on 26/30 (significant) |
| big outage | +141 €/day, better on 16/30 (not significant) | +384 €/day, better on 22/30 (significant) |
| complaint storm | +315 €/day, better on 22/30 (significant) | −83 €/day, better on 10/30 (not significant) |
| quiet day | +344 €/day, better on 30/30 (significant) | −549 €/day, better on 0/30 (significant) |

### Service

| Scenario | SLA (20 s): Erlang C / reactive / margin / RL | Hang-ups: Erlang C / reactive / margin / RL |
|---|---|---|
| normal | 90.5% / 91.8% / 97.2% / 95.5% | 3.3% / 2.8% / 1.1% / 1.6% |
| network outage | 81.5% / 85.8% / 92.2% / 91.0% | 7.0% / 4.8% / 2.7% / 3.1% |
| billing day | 78.4% / 84.1% / 92.4% / 90.6% | 7.9% / 5.4% / 3.1% / 3.3% |
| afternoon promo | 78.6% / 84.2% / 93.6% / 90.3% | 8.0% / 5.3% / 2.1% / 3.2% |
| late outage | 84.2% / 87.6% / 92.1% / 93.0% | 6.0% / 4.4% / 3.0% / 2.5% |
| big outage | 75.2% / 81.1% / 85.7% / 86.1% | 10.4% / 6.4% / 5.2% / 4.9% |
| complaint storm | 84.2% / 87.3% / 90.9% / 90.6% | 6.6% / 4.6% / 4.0% / 3.6% |
| quiet day | 97.5% / 97.7% / 99.3% / 98.3% | 0.8% / 0.8% / 0.3% / 0.7% |

The Erlang C plan misses the 80% target on billing day, afternoon promo and big outage; every other policy meets it everywhere.

### What this shows

- **The RL agent beats the classic Erlang C plan** on 7 of 8 scenarios (by 194 to 2,686 €/day), and the reactive rule on 5 of 8.
- **No policy wins everywhere.** On average, the margin rule and the RL agent are tied (8,827 € vs 8,864 €/day). The margin rule is better when the extra calls are general calls (a fixed +4 general advisors is exactly right there); the RL agent is better when the surprise is elsewhere or at an unusual time (late outage, big outage, complaint storm) and on the quiet day.
- **The RL agent is the most robust:** on every scenario it is at most 717 €/day from the best policy, the smallest worst case of the four. A fixed margin wastes money when calls are fewer than expected (quiet day: +900 € vs Erlang C) and is not always enough when the surprise hits another queue.
- **It really reacts** (measured from 10h to 13h, compared with the Erlang C plan): the 3 agents add +1.0 to +1.7 advisors on a normal day, +3.1 to +4.6 during the outage (mostly on the technical queue), and 0 to +1.3 on the quiet day.
- **The 3 agents agree closely** (small ± spread), so the result does not depend on one lucky training.
- On the quiet day, the simple Erlang C plan is still the cheapest: the agent keeps a small safety margin it does not need.

### What drives the agent's decisions (XAI)

Permutation importance (`python -m workforce.explain`, chart `what_matters.html`): each input is shuffled across about 1,000 situations and we measure how much the decisions change. The most important inputs are, in order: **complaint calls received vs forecast**, **priority on**, **technical calls received vs forecast**, **time of day**, **general calls received vs forecast**. The agent watches the calls coming in, not the clock. The number of customers waiting at the decision moment matters very little: it is usually 0 or 1, because customers are either answered quickly or hang up.

The charts in `workforce/models/explain/`: `timeline_<scenario>.html` (advisors on duty for each policy every 15 minutes, with calls received ÷ forecast), `corrections.html` (advisors added or removed vs the plan, per hour and queue), `what_matters.html`.

## How we got here (useful for the report)

1. **The price of bad service decides what the agent learns.** With cheap prices (8 € per hang-up, 2 € per late answer), the first agent cut staff by a third and let service collapse (SLA 71%, 13% hang-ups): it did exactly what the reward asked. With 30 € / 5 €, the gain was not significant; with 50 € / 8 €, significant. We kept 50 € / 8 €.
2. **The first agent only learned a margin.** Trained on 4 fixed scenarios and unable to see incoming calls, it added roughly the same advisors every day. The XAI timeline showed it; a simple "Erlang C + fixed margin" rule then did as well as the agent. That rule became the margin rule.
3. **The fix:** the agent now sees calls received vs forecast, and trains on random surprise days, so a fixed margin is no longer enough. It now reacts, as shown above.

## Limits (to say in the report)

- Call volumes, durations, patience and the 25 €/hour cost are **placeholders**, to calibrate on public call center data (Technion) and on statistics from real LangGraph calls.
- Satisfaction is a placeholder formula based on waiting time, until it comes from the client agent.
- The result depends on the reward prices (50 € / 8 €), a business choice the team should agree on.
- The agent changes its requests often (it calls advisors in and sends them home after the 1-hour minimum more often than a real manager would; 4 to 7 queue switches per day). A small cost per staffing change in the reward would make it steadier.
- The test scenarios are invented; real peaks could be modeled from incident logs.
