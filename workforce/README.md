# Workforce manager

The floor manager decides, every 15 minutes, how many advisors work on each queue (general, technical, complaint). It is trained with reinforcement learning (PPO) and compared with three classic rules on days it never saw.

## Files

| File | What it does |
|---|---|
| `simulation.py` | The simulated call center (SimPy): one day of calls, three queues, 25 advisors with skills, customers who hang up, realistic staffing rules, test scenarios, and the KPIs. |
| `calibration.json` | Every calibrated value with its source: arrivals, call durations, patience, forecast error, roster, and the tuned rule settings. |
| `manager.py` | The rules: Erlang C plan, margin rule, reactive rule. |
| `tune_rules.py` | Tunes the reactive and margin rules for the calibrated center (stored in `calibration.json`). |
| `env.py` | The simulation as a Gymnasium environment, random training days, the reward, and `RLPolicy` to use a trained agent like any rule. |
| `train.py` | Trains an agent with PPO (Stable-Baselines3). |
| `compare.py` | Compares all policies on 8 test scenarios, 30 days each, with a statistical test. |
| `explain.py` | Charts that explain the agent's decisions (XAI). |
| `../scripts/calibrate_workforce.py` | Fits the calibration from the Technion files, roster included. |
| `../scripts/run_workforce.py` | Reruns everything in order: calibration, rule tuning, training, comparison, charts. |

## How to run

```bash
pip install -e ".[dev,rl]"            # RL packages, including PyTorch (about 1 GB) and Plotly
python scripts/run_workforce.py       # everything below in one go (about 1 hour 15 min)
                                      # add --skip-calibration if you do not have the Technion files

python -m workforce.simulation        # one day with the Erlang C plan
python -m workforce.tune_rules        # tune the reactive and margin rules (a few minutes)
python -m workforce.train --seed 0    # about 20 minutes on a laptop; repeat with --seed 1 and --seed 2
python -m workforce.compare           # comparison tables: workforce/models/comparison.json and results.md
python -m workforce.explain           # the XAI charts (workforce/models/explain/*.html)
```

## Calibration

The simulation's numbers come from real call center data: the Technion "Anonymous Bank" call center, every call of 1999 (444,448 records). Phase 1 used the published analysis of this center (Brown et al., 2005); phase 2 fits the values from the raw files with `scripts/calibrate_workforce.py`. Every value and its source are in [`calibration.json`](./calibration.json).

- **Arrivals per hour:** queue entries on the bank's working days, **Sunday to Thursday** (257 days), grouped by hour from 8h to 20h, then rescaled to CallVerse's volume of about 1,050 calls a day. Dates in the files are written `990901` (1 September 1999); the bank closes at 14:00 on Friday and reopens around 20:00 on Saturday, so those days are left out.
- **Call durations:** log-normal fits of served (`AGENT`) calls after removing durations under 10 seconds: 3.2 minutes for regular calls (general), 6.9 minutes for internet calls (technical). Complaints use the longest observed type, a temporary assumption until phase 3.
- **Patience:** censored exponential estimate, all queue waiting time divided by all hang-ups: about 8.0 minutes on average, by hour of the day.
- **Forecast error:** each 15-minute period is up to ±20% around the forecast (Brown et al., 2005).
- **Roster:** sized by the script so the Erlang C plan uses at most 80% of it at its busiest hour, per skill: 25 advisors (Erlang C peak: 20).
- **Rules:** the reactive and margin rules are tuned for this calibrated center by `python -m workforce.tune_rules`, on random training days (never the test days).

```bash
python scripts/calibrate_workforce.py                 # needs the 12 files in data/raw/technion/
python scripts/calibrate_workforce.py --roster-only   # only resize the roster, no raw files needed
```

The raw files are not on GitHub: see `data/raw/README.md` for where to download them.

## The simulated call center

- About 1,050 calls a day (8h–20h), using the rescaled weekday arrival curve fitted from the Technion data.
- 25 advisors: 8 general only, 10 general + technical, 7 general + complaint. Paid 25 €/hour when on duty.
- Call durations (3.2 min general, 6.9 min technical and complaint) and patience (about 8 min) are fitted from the Technion data; `calibration.json` has the exact values and observation counts.
- Each 15-minute arrival period varies randomly by up to 20% around the forecast, matching the published forecast-error range.
- **Realistic staffing rules**, the same for every policy: an advisor called in stays **at least 1 hour**; moving to another queue takes **2 minutes**; a policy can turn on **priority** (multi-skilled advisors on general calls serve waiting technical or complaint calls first).

## The policies

| Policy | What it does |
|---|---|
| **Erlang C plan** | Staffing computed in advance from the forecast with the Erlang C formula (80% of calls answered within 20 s), hour by hour. Never changes during the day. |
| **Reactive rule** | The Erlang C plan, plus one advisor per waiting customer (at most 5 per queue). |
| **Margin rule** | The Erlang C plan, plus a fixed safety margin (after tuning: 4 general and 1 technical advisors), priority on. Found by analyzing what an earlier RL agent did (see "How we got here"). |
| **RL agent (PPO)** | Every 15 minutes, corrects the Erlang C plan by −3 to +3 advisors per queue and turns priority on or off. |

The two rules' settings (and whether to use priority) are tuned for the calibrated center on random training days, never on the test days (`workforce.tune_rules`).

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

Calibrated simulation (Technion data, 25 advisors). Same 30 test days per scenario for every policy. RL = average of 3 agents ± the spread between them.

### Total cost per day (staff + price of bad service)

| Scenario | Erlang C plan | Reactive rule | Margin rule | RL agent | Best |
|---|---|---|---|---|---|
| normal | 6,375 € | 6,089 € | 6,151 € | **6,055 € ±117** | RL agent |
| network outage | 8,887 € | **7,403 €** | 7,542 € | 7,613 € ±183 | Reactive rule |
| billing day | 9,579 € | 7,950 € | **7,243 €** | 7,289 € ±327 | Margin rule |
| afternoon promo | 9,189 € | 7,185 € | **6,343 €** | 6,913 € ±35 | Margin rule |
| late outage | 7,614 € | 6,688 € | 6,864 € | **6,676 € ±105** | RL agent |
| big outage | 12,013 € | 9,581 € | 9,606 € | **9,469 € ±279** | RL agent |
| complaint storm | 8,609 € | 7,036 € | 7,696 € | **7,026 € ±97** | RL agent |
| quiet day | 4,588 € | **4,587 €** | 5,784 € | 4,981 € ±162 | Reactive rule |
| **average** | 8,357 € | 7,065 € | 7,154 € | 7,003 € | |
| **worst gap to the best policy** | 2,846 € | 842 € | 1,198 € | 570 € | |

### Is the RL agent really better? (paired test on the same days; + = RL cheaper)

| Scenario | RL vs Erlang C | RL vs reactive rule | RL vs margin rule |
|---|---|---|---|
| normal | +320 €/day, better on 21/30 (significant) | +34 €/day, better on 17/30 (not significant) | +97 €/day, better on 19/30 (significant) |
| network outage | +1,273 €/day, better on 29/30 (significant) | -210 €/day, better on 10/30 (not significant) | -71 €/day, better on 11/30 (not significant) |
| billing day | +2,289 €/day, better on 30/30 (significant) | +660 €/day, better on 25/30 (significant) | -46 €/day, better on 15/30 (not significant) |
| afternoon promo | +2,276 €/day, better on 29/30 (significant) | +272 €/day, better on 20/30 (significant) | -570 €/day, better on 3/30 (significant) |
| late outage | +939 €/day, better on 28/30 (significant) | +12 €/day, better on 13/30 (not significant) | +188 €/day, better on 17/30 (not significant) |
| big outage | +2,545 €/day, better on 30/30 (significant) | +112 €/day, better on 19/30 (not significant) | +137 €/day, better on 17/30 (not significant) |
| complaint storm | +1,583 €/day, better on 29/30 (significant) | +10 €/day, better on 12/30 (not significant) | +670 €/day, better on 25/30 (significant) |
| quiet day | -393 €/day, better on 2/30 (significant) | -395 €/day, better on 0/30 (significant) | +803 €/day, better on 30/30 (significant) |

### Service

| Scenario | SLA (20 s): Erlang C / reactive / margin / RL | Hang-ups: Erlang C / reactive / margin / RL |
|---|---|---|
| normal | 87.9% / 90.3% / 97.8% / 94.5% | 2.5% / 1.8% / 0.6% / 1.1% |
| network outage | 79.2% / 85.5% / 91.8% / 88.7% | 5.4% / 3.0% / 2.1% / 2.6% |
| billing day | 77.1% / 84.8% / 94.1% / 91.2% | 5.2% / 2.8% / 1.6% / 1.7% |
| afternoon promo | 76.4% / 85.7% / 96.6% / 90.7% | 5.4% / 2.4% / 0.6% / 1.7% |
| late outage | 84.0% / 88.3% / 94.9% / 92.6% | 4.0% / 2.2% / 1.4% / 1.6% |
| big outage | 71.6% / 78.8% / 84.7% / 82.8% | 9.3% / 5.4% / 4.5% / 4.7% |
| complaint storm | 81.1% / 87.3% / 92.0% / 91.1% | 5.5% / 2.7% / 2.5% / 2.1% |
| quiet day | 96.7% / 97.1% / 99.4% / 97.5% | 0.6% / 0.5% / 0.2% / 0.5% |

The Erlang C plan misses the 80% SLA target on 4 scenarios (network outage, billing day, afternoon promo, big outage) and the reactive rule on 1 (big outage); the margin rule and the RL agent meet it on all 8.

### What this shows

- **The RL agent clearly beats the classic Erlang C plan:** cheaper on 7 of 8 scenarios, by 320 to 2,545 €/day, significantly every time; only on the quiet day is the plan cheaper (by 393 €/day).
- **On average it is the cheapest policy, but by a small margin:** 7,003 €/day against 7,065 € for the reactive rule and 7,154 € for the margin rule. Against the two tuned rules, most differences are not significant: a well-tuned simple rule is a strong opponent.
- **It is the most robust:** on every scenario it is at most 570 €/day from the best policy, the smallest worst case of the four (reactive rule 842 €, margin rule 1,198 €). Each rule has a weak spot: the margin rule pays for idle staff on the quiet day (+1,197 €), the reactive rule reacts late to general-call surges (afternoon promo +842 €).
- **Where it wins and loses:** best on 4 of 8 scenarios (normal, late outage, big outage, complaint storm). The margin rule is best when the extra calls are general calls (billing day, afternoon promo), because its fixed general margin is exactly right there; the simple rules are cheaper on the quiet day, where the agent keeps a safety margin it does not need.
- **The 3 agents agree** (spread of 35 to 327 €/day), so the result does not depend on one lucky training.

### What drives the agent's decisions (XAI)

Permutation importance (`python -m workforce.explain`, chart `what_matters.html`): each input is shuffled across many situations and we measure how much the decisions change. The most important inputs are, in order: **complaint calls received vs forecast** (0.99), **technical calls received vs forecast** (0.77), **time of day** (0.77), **general calls received vs forecast** (0.72), **priority on** (0.71) and **forecast for the next hour** (0.69). The agent mainly watches the calls coming in compared with what was expected, which is how it notices a surprise without being told about it.

The charts in `workforce/models/explain/`: `timeline_<scenario>.html` (advisors on duty for each policy every 15 minutes, with calls received ÷ forecast), `corrections.html` (advisors added or removed vs the plan, per hour and queue), `what_matters.html`.

## How we got here (useful for the report)

1. **The price of bad service decides what the agent learns.** With cheap prices (8 € per hang-up, 2 € per late answer), the first agent cut staff by a third and let service collapse (SLA 71%, 13% hang-ups): it did exactly what the reward asked. With 30 € / 5 €, the gain was not significant; with 50 € / 8 €, significant. We kept 50 € / 8 €.
2. **The first agent only learned a margin.** Trained on 4 fixed scenarios and unable to see incoming calls, it added roughly the same advisors every day. The XAI timeline showed it; a simple "Erlang C + fixed margin" rule then did as well as the agent. That rule became the margin rule.
3. **The fix:** the agent now sees calls received vs forecast, and trains on random surprise days, so a fixed margin is no longer enough. It now reacts.
4. **From invented to real numbers.** The first simulation used plausible but invented values. It is now calibrated on 444,448 real calls. A review of the first calibration found that dates were read as 2099 and the wrong days of the week were used, and that the roster had no room left for surprises (the Erlang C plan needed 20 of 21 advisors at peak). After fixing both, the reactive rule became much stronger (it finally had advisors to call in), and the comparison above was rerun from scratch.

## Limits (to say in the report)

- The center is calibrated on a **bank** call center from 1999, not a telecom operator; the complaint duration is an assumption (the longest bank call type). Phase 3 will use call durations from CallVerse's own LangGraph calls.
- The daily volume of 1,050 calls, the call mix (55/30/15%) and the 25 €/hour cost are team choices, not measured values.
- Satisfaction is a placeholder formula based on waiting time, until it comes from the client agent.
- The result depends on the reward prices (50 € / 8 €), a business choice the team should agree on.
- The agent changes its requests often (it calls advisors in and sends them home after the 1-hour minimum more often than a real manager would; 4 to 7 queue switches per day). A small cost per staffing change in the reward would make it steadier.
- The test scenarios are invented; real peaks could be modeled from incident logs.
