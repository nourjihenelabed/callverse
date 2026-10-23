# Workforce Manager Guide

This document explains the Workforce Manager in simple terms: what it does,
how the simulation works, what each file contains, and what happened in the
three calibration phases.

## 1. The goal of the Workforce Manager

The Workforce Manager decides how many advisors should work on each queue
during a simulated call-center day.

CallVerse has three queues:

- `general`: normal customer questions
- `technical`: internet or technical problems
- `complaint`: unhappy customers

The center is open from 08:00 to 20:00. Every 15 minutes, a workforce policy
decides how many advisors should work on each queue.

The manager tries to balance two opposite goals:

1. Keep waiting times and abandoned calls low.
2. Avoid paying for more advisors than necessary.

## 2. The main workforce files

| File | Purpose |
|---|---|
| `simulation.py` | Runs one simulated call-center day. It creates calls, queues, advisors, waits, service times, abandonments, and KPIs. |
| `manager.py` | Contains the traditional staffing policies and the Erlang C calculation. |
| `env.py` | Converts the call-center simulation into a Gymnasium environment for reinforcement learning. |
| `train.py` | Trains PPO reinforcement-learning agents. |
| `compare.py` | Compares the Erlang C, reactive, margin, and RL policies on the same test days. |
| `explain.py` | Creates charts explaining when and why the RL agent changes staffing. |
| `calibration.json` | Stores the current data-based workforce values. |
| `scripts/calibrate_workforce.py` | Reads the Technion raw files and produces `calibration.json`. |
| `tests/test_workforce.py` | Tests the simulation and traditional workforce rules. |
| `tests/test_workforce_rl.py` | Tests the Gymnasium environment and short RL training. |

## 3. How one simulated day works

`CallCenter` in `simulation.py` runs one day using SimPy.

### Step 1: Calls arrive

Calls arrive at different rates during the day. The arrival curve normally has
busy and quiet hours.

Each call is assigned to a queue using the configured call mix. The policy
does not control when calls arrive.

### Step 2: Customers wait

If no suitable advisor is free, the call enters its queue.

The customer receives a random patience time. If the customer waits longer than
that time, the call is abandoned.

### Step 3: Advisors answer calls

Advisors have skills:

- Some can work only on general calls.
- Some can work on general and technical calls.
- Some can work on general and complaint calls.

An advisor can work on one queue at a time. Moving to another queue takes setup
time, and a called-in advisor stays on duty for at least one hour.

### Step 4: Calls receive service

The service duration is sampled from a log-normal distribution. The mean and
spread come from `calibration.json`.

### Step 5: The policy makes decisions

Every 15 minutes, the policy sees a `Snapshot` containing information such as:

- Current waiting customers
- Advisors working on each queue
- Recent average waiting time
- Recent SLA result
- Recent abandoned calls
- Calls received compared with the forecast
- Forecast calls for the next hour

The policy returns a request such as:

```python
{
    "general": 8,
    "technical": 5,
    "complaint": 3,
    "priority": True,
}
```

### Step 6: KPIs are calculated

At the end of the day, the simulation records:

- Offered calls
- Answered calls
- Abandonment rate
- SLA percentage
- Average waiting time
- Advisor occupancy
- Staff cost
- Satisfaction
- Queue switches

## 4. The four workforce policies

### 4.1 Erlang C plan

The Erlang C plan is the traditional baseline.

It calculates how many advisors are needed in each hour to reach the target:

- At least 80% of calls answered
- Within 20 seconds

The calculation uses:

- Calls per hour
- Average call duration
- SLA target
- Number of advisors

It creates an hourly plan and does not react to unexpected calls during the
day.

### 4.2 Reactive rule

The reactive rule starts with the Erlang C plan.

When customers are waiting, it adds advisors to the affected queue, up to a
maximum extra amount. It is a simple rule-based reaction to the current queue.

### 4.3 Margin rule

The margin rule starts with Erlang C and adds a fixed staffing margin,
especially to the general queue. It also enables priority handling.

Priority allows multi-skilled advisors serving general calls to help waiting
technical or complaint calls.

### 4.4 RL agent

The RL agent learns from repeated simulated days.

It starts from the Erlang C plan and chooses:

- A correction between -3 and +3 advisors for each queue.
- Whether priority should be on or off.

The RL agent receives a negative reward based on cost:

- Advisor staffing cost
- Penalty for abandoned calls
- Penalty for calls answered after 20 seconds

The agent therefore learns that both overstaffing and poor service are
expensive.

## 5. The RL environment

`env.py` wraps the simulation as a Gymnasium environment.

One episode is one simulated day.

There are 52 decisions in a normal day because:

```text
12 hours x 4 decisions per hour = 48 decisions
```

The simulation also includes the closing and overtime periods, which is why
the environment's exact step behavior is tested in
`tests/test_workforce_rl.py`.

The observation contains normalized values such as:

- Time of day
- Waiting customers per queue
- Advisors working per queue
- Recent average wait
- Erlang C plan
- Recent SLA
- Recent abandoned calls
- Next-hour forecast
- Calls received compared with forecast
- Priority state

The action contains four values:

```text
general correction
technical correction
complaint correction
priority on/off
```

## 6. Calibration: why it is needed

The original simulation used invented placeholder numbers. For example, it
used longer service times and shorter customer patience than the real bank
data suggested.

Calibration replaces those invented statistical inputs with traceable values.

The simulation method and policies do not change. The main change is where the
numbers come from.

## 7. Phase 1: published values

Phase 1 used summary values from Brown et al. (2005), based on the Technion
Anonymous Bank call center.

Phase 1 changed:

- General call duration to about 3 minutes.
- Technical call duration to about 6.7 minutes.
- Complaint duration to the longest observed type as a temporary proxy.
- Average patience to about 7.4 minutes.
- Forecast variation to up to about +/-20% per 15-minute period.
- The roster from 30 to 21 advisors.

The roster became:

- 6 general-only advisors
- 9 general and technical advisors
- 6 general and complaint advisors

The sources were recorded in `calibration.json`, including paper page and
table/section references.

The three RL agents were retrained with seeds 0, 1, and 2. The comparison and
explanation charts were regenerated.

## 8. Phase 2: raw Technion data

Phase 2 builds on Phase 1. It does not replace the simulation design.

Instead of using only published summary numbers, Phase 2 reads the original
call-by-call files from the Technion Anonymous Bank archive.

The archive contains twelve monthly files from 1999. The downloaded files are
stored locally in:

```text
data/raw/technion/
```

They are ignored by Git because they are raw external data.

The Phase 2 script is:

```text
scripts/calibrate_workforce.py
```

Run it with:

```powershell
py scripts\calibrate_workforce.py
```

The script reads the documented fields:

- `type`
- `date`
- `q_start`
- `q_time`
- `outcome`
- `ser_time`

It currently reads 444,448 records from the twelve files.

### Phase 2 call mapping

The raw bank data uses different service-type codes. The script maps them to
CallVerse queues:

```text
PS, PE -> general
IN     -> technical
NE     -> complaint
```

The raw bank data has no exact CallVerse complaint category. Therefore, the
complaint duration uses the longest observed service type as a temporary
proxy. This assumption remains documented until Phase 3.

### Phase 2 calculations

The script calculates:

1. Weekday arrival volume for every hour.
2. Daily volume variation.
3. Mean service duration by queue.
4. Log-normal service spread by queue.
5. Hourly patience using queue-time exposure and abandoned calls.

Service calls shorter than 10 seconds are excluded, following the calibration
plan and the Brown et al. treatment of very short calls.

The weekday arrival curve is rescaled to about 1,050 calls per day. This is a
CallVerse team choice, not a claim that the bank had exactly that volume.

The script writes the results to:

```text
workforce/calibration.json
```

`CenterConfig` reads that file when the simulation starts.

Phase 2 also keeps some Phase 1/team decisions:

- The +/-20% forecast factor.
- The 21-advisor roster.
- The advisor cost.
- The reward penalties.
- The CallVerse opening hours and daily-volume scale.

After recalibrating, the three RL agents must be retrained and the comparison
and XAI outputs must be regenerated.

## 9. Phase 3: CallVerse AI-call statistics

Phase 3 is planned for later.

It requires the CallVerse agents to produce representative end-to-end calls,
especially:

- Client simulator
- Advisor
- Complaint specialist
- Working LangGraph flow

The team would run several hundred calls and measure:

- Queue
- Customer persona
- Conversation turns
- Tool lookups
- Outcome
- Satisfaction
- Complaint severity where relevant

Turns and tool lookups would be converted into simulated minutes using a rule
agreed by the team. Those AI-call durations would then replace the bank-based
durations for CallVerse telecom requests.

The Technion data would still be useful for:

- Arrival patterns
- Customer waiting and abandonment behavior

Phase 3 should not be treated as complete while the agents are still simple
stubs or while their behavior is not representative of the final project.

## 10. What is currently complete

The current Phase 2 branch contains:

- Phase 2 calibration script.
- Phase 2 fitted `calibration.json`.
- Phase 2 runtime loading in `simulation.py`.
- Updated workforce documentation.
- Updated raw-data documentation.
- Updated tests.
- Three retrained RL agents locally.
- Regenerated comparison and XAI outputs locally.

The generated model files and reports are ignored by Git and can be rebuilt.

The final test result after the Phase 2 implementation is:

```text
42 passed, 2 skipped
```

## 11. Recommended workflow

Before committing Phase 2:

```powershell
py scripts\calibrate_workforce.py
py -m pytest
```

Then review the changes:

```powershell
git status
git diff --check
```

The raw files in `data/raw/technion/` should not be added to Git.

After Phase 2 is reviewed and merged, Phase 3 can be started later on a new
branch when the other CallVerse agents are ready.
