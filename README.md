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
| `callverse/knowledge.py` | The one place where workers search the company rules |
| `callverse/config.py` | Reads `settings.yaml` and your `.env` file |
| `callverse/agents/` | One folder per worker |
| `workforce/` | The pretend call center and the floor manager |
| `dashboard/` | The scoreboard |
| `data/policies/` | Company rules, written as text files |
| `data/raw/` | Downloaded public datasets. Not on GitHub because they are too big. |
| `scripts/` | Small programs that prepare the data and fill the database |
| `.env.example` | A model for your own `.env` file, which holds your database password |
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
7. Ask the team before changing `state.py`, `llm.py`, `graph.py`, or `knowledge.py`, because everyone depends on them.

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

### 1. The database

We store customers, complaints, and tickets in PostgreSQL.

1. Install PostgreSQL from postgresql.org/download. Remember the password you choose during the install.
2. Open "SQL Shell (psql)", log in with that password, and type: `CREATE DATABASE callverse;`
3. In the project folder, copy `.env.example` to a new file named `.env`.
4. In `.env`, replace `YOUR_PASSWORD` with your PostgreSQL password.

Never put your `.env` file on GitHub. It is already blocked, so `git add .` will skip it.

### 2. Python and the project

1. Install Python 3.11 or newer from python.org.
2. In the project folder, create and turn on your own Python space:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
   ```

   On Windows, if `python` is blocked or points to another program, use `py -m venv .venv` instead.
3. Install the project and its packages:

   ```bash
   pip install -e ".[dev]"
   ```

4. Check that a full pretend call works:

   ```bash
   pytest
   ```

   This needs no AI model and no database. Run it before every pull request.

### 3. The AI model

1. Install Ollama from ollama.com/download. It needs about 15 GB of free disk space with the models.
2. Download the model the team picked (written in `settings.yaml`), for example `ollama pull <model>`.
3. If your computer is too small, a teammate can run Ollama for you: put their address in your `.env` as `OLLAMA_BASE_URL`.

### 4. The company rules search

1. Install Chroma: `pip install -e ".[dev,rag]"`. If Windows blocks it, set `rag: store: "local"` in your own copy of `settings.yaml` (don't push that change) and skip this step.
2. With Ollama running, build the index: `python scripts/build_index.py "Combien coûte la Box Fibre ?"`
3. Build it again whenever a file in `data/policies/` changes.

### 5. Try the advisor

With Ollama running, the database filled and the rules indexed:

```bash
python scripts/try_advisor.py                  # 4 prepared questions
python scripts/try_advisor.py --chat           # type your own messages
```

It shows what the advisor understood, the tools it used and its answer. See `callverse/agents/advisor/README.md`.

### 6. Run whole pretend calls

The AI customer talks to the AI advisor from start to finish:

```bash
python scripts/try_call.py                                         # 3 random calls
python scripts/try_call.py --reason complaint_billing --persona demanding
```

See `callverse/agents/client/README.md`. Complaint calls open real tickets: run `python scripts/seed_data.py` again to clean them.
