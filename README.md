# CallVerse

CallVerse is a pretend call center that lives inside a computer.

Every worker in it is a computer program. Pretend customers call in, the programs answer them, and we watch how well the call center does. Then we try to make it better.

This is a school project built by a team of 7 people.

## The workers

| Worker | What it does |
|---|---|
| Pretend customer | Plays the person who calls. Some are calm, some are in a hurry, some are angry, some want to leave. |
| Advisor | Picks up the phone and answers everyday questions. It is the only worker the customer talks to. |
| Complaint specialist | Helps the advisor when a customer is unhappy. It finds out what went wrong and suggests a fair fix that follows the company rules. Very serious cases go to a real boss. |
| Reviewer | Listens to each call afterwards and gives it a grade, with tips to do better. |
| Floor manager | Watches how long people wait and moves workers around. It gets better by practicing many pretend days. |

All the workers read from the same shared folder: the company rules, common questions, and each customer's past problems.

## How one call works

1. A pretend customer calls and waits in line.
2. The advisor answers.
3. If it is a complaint, the advisor asks the complaint specialist for help, then gives the customer the fix.
4. When the call ends, the reviewer grades it.
5. The floor manager looks at how busy the call center is and moves workers if needed.
6. A scoreboard shows the results: waiting time, problems solved, and call grades.

## What is in this repository

This is the plan. Folders will appear as we build them.

| Folder or file | What it holds |
|---|---|
| `settings.yaml` | Shared settings, like which AI model we all use |
| `callverse/state.py` | The "case file" that travels through a call. Everyone agrees on it before changing it. |
| `callverse/llm.py` | The one place where the AI model is set up. Everyone uses it. |
| `callverse/graph.py` | Decides which worker speaks next during a call |
| `callverse/agents/` | One folder per worker |
| `workforce/` | The pretend call center and the floor manager |
| `dashboard/` | The scoreboard |
| `data/` | Company rules and pretend customer data |
| `scripts/` | Small programs that prepare the data |
| `tests/` | Checks that one full call still works |

## The team

| Name | Works on |
|---|---|
|  | Pretend customer |
|  | Advisor |
|  | Complaint specialist |
|  | Reviewer |
|  | Floor manager |
|  | Scoreboard |
|  | Shared files and data |

## How we work together

1. Never change `main` directly. `main` is the official version and it must always work.
2. For each task, make your own branch from `main`. Name it `worker/task`, for example `complaint/classify`.
3. Mostly change files in your own folder.
4. When your task works, open a pull request. One teammate checks it.
5. Click "Squash and merge", then delete the branch.
6. Start your next task from a fresh `main`.
7. Ask the team before changing `state.py`, `llm.py`, or `graph.py`, because everyone depends on them.

## Everyday commands

Start a new task:

```bash
git checkout main
git pull
git checkout -b complaint/classify
```

Save your work:

```bash
git add .
git commit -m "say what you changed"
```

Send your work to GitHub:

```bash
git push -u origin complaint/classify
```

After your pull request is merged:

```bash
git checkout main
git pull
git branch -D complaint/classify
```

Get your teammates' latest work while you are still on your branch:

```bash
git pull origin main
```

## Setting up on your computer

Coming soon. We will add the steps once the team picks the AI model.
