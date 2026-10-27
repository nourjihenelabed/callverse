"""Is the AI reviewer trustworthy? Compares its grades with a person's grades on the same calls.

Two steps, run from the project folder:

1. Export calls to grade (the AI's grades are NOT shown, so they cannot influence you):
       python scripts/judge_agreement.py export --run night-2 --calls 10
   It writes two files in evaluation/:
   - calls_to_grade.html : the conversations, easy to read in a browser
   - human_grades.csv    : one line per call; fill in relevance, accuracy, compliance, empathy (1 to 5)
     The file opens in Excel; save it as CSV (keep the ; separator).

2. Compare your grades with the AI reviewer's:
       python scripts/judge_agreement.py compare
   It prints, for each criterion: exact agreement, agreement within 1 point, the average difference
   (+ = the AI is more generous than you) and Cohen's weighted kappa, and writes evaluation/judge_agreement.md.

Only the 4 criteria decided by reading the call are compared: resolution and satisfaction come straight
from how the call ended, so there is nothing to judge.
"""

import argparse
import csv
import html
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.results import load_calls  # noqa: E402

CRITERIA = ("relevance", "accuracy", "compliance", "empathy")
FOLDER = Path(__file__).resolve().parent.parent / "evaluation"
CSV_PATH = FOLDER / "human_grades.csv"
HTML_PATH = FOLDER / "calls_to_grade.html"
REPORT_PATH = FOLDER / "judge_agreement.md"

GUIDE = {
    "relevance": "Did the advisor answer what the customer asked, without going off topic?",
    "accuracy": "Is what the advisor said true, according to the catalogue, the customer's file and the rules?",
    "compliance": "Did the advisor follow the company guide (look things up first, never mention the churn risk, "
                  "no password asked, complaints to the specialist, decisions announced)?",
    "empathy": "Was the tone right for this customer's mood?",
}


# ---------------------------------------------------------------- 1. export

def pick(rows: list[dict], count: int, seed: int = 7) -> list[dict]:
    """A varied sample: calls of every outcome, then random ones, in a shuffled order."""
    rng = random.Random(seed)
    groups = {}
    for r in rows:
        groups.setdefault(r["outcome"], []).append(r)
    chosen = []
    while len(chosen) < min(count, len(rows)):
        for outcome in sorted(groups):
            left = [r for r in groups[outcome] if r not in chosen]
            if left and len(chosen) < count:
                chosen.append(rng.choice(left))
    rng.shuffle(chosen)
    return chosen


def export(rows: list[dict], folder: Path = FOLDER) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, html_path = folder / CSV_PATH.name, folder / HTML_PATH.name
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Excel shows the accents
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["call_id", "reason", "mood", *CRITERIA, "comment"])
        for r in rows:
            writer.writerow([r["call_id"], r["call_reason"], r["persona"], "", "", "", "", ""])

    blocks = []
    for n, r in enumerate(rows, 1):
        lines = "".join(f'<p class="{esc(m["role"])}"><b>{"Client" if m["role"] == "client" else "Conseiller"} :</b> '
                        f'{esc(m["content"])}</p>' for m in r["messages"])
        decision = ""
        if r.get("is_complaint"):
            decision = (f'<p class="note">Complaint specialist: {esc(r["complaint_category"])}, severity '
                        f'{esc(r["complaint_severity"])}, compensation {(r["compensation_cents"] or 0) / 100:.2f} €, '
                        f'supervisor: {"yes" if r["escalated"] else "no"}.</p>')
        if r.get("retention_offer"):
            decision += f'<p class="note">Supervisor\'s loyalty offer: {esc(r["retention_offer"])}.</p>'
        blocks.append(f'<section><h2>{n}. {esc(r["call_id"])}</h2><p class="note">Customer {esc(r["customer_id"])}, '
                      f'reason {esc(r["call_reason"])}, mood {esc(r["persona"])}, ended: {esc(r["outcome"])}.</p>'
                      f'{decision}{lines}</section>')
    guide = "".join(f"<li><b>{c}</b>: {esc(q)}</li>" for c, q in GUIDE.items())
    html_path.write_text(
        '<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width, initial-scale=1"><title>Calls to grade</title><style>'
        "body{font-family:system-ui,sans-serif;max-width:900px;margin:auto;padding:20px;color:#1f2933}"
        "section{border:1px solid #e4e7eb;border-radius:8px;padding:12px 16px;margin:16px 0}"
        ".client{background:#f3f4f6;padding:6px 10px;border-radius:6px}"
        ".advisor{background:#eaf1fb;padding:6px 10px;border-radius:6px;margin-left:32px}"
        ".note{color:#52606d;font-size:14px}</style></head><body>"
        f"<h1>Calls to grade</h1><p>Grade each call from 1 (very bad) to 5 (excellent) in "
        f"<code>{esc(csv_path.name)}</code>. Grade the <b>advisor</b>, not the customer.</p><ul>{guide}</ul>"
        f"{''.join(blocks)}</body></html>", encoding="utf-8")
    return csv_path, html_path


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


# ---------------------------------------------------------------- 2. compare

def read_grades(path: Path = CSV_PATH) -> dict[str, dict[str, int]]:
    """{call_id: {criterion: grade}} from the filled CSV (accepts ; or , as separator)."""
    text = path.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=";,")
    grades = {}
    for line in csv.DictReader(text.splitlines(), dialect=dialect):
        values = {c: int(line[c]) for c in CRITERIA if (line.get(c) or "").strip()}
        if values:
            grades[line["call_id"].strip()] = values
    return grades


def weighted_kappa(a: list[int], b: list[int], levels=range(1, 6)) -> float | None:
    """Cohen's kappa with quadratic weights: 1 = perfect agreement, 0 = no better than chance."""
    levels = list(levels)
    n = len(a)
    if n < 2:
        return None
    k = len(levels)
    index = {v: i for i, v in enumerate(levels)}
    weight = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1 / n
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    disagreement = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k))
    expected = sum(weight[i][j] * row[i] * col[j] for i in range(k) for j in range(k))
    return None if expected == 0 else round(1 - disagreement / expected, 2)


def agreement(human: dict[str, dict[str, int]], rows: list[dict]) -> dict:
    by_id = {r["call_id"]: r for r in rows}
    result = {}
    for c in CRITERIA:
        pairs = [(g[c], by_id[cid]["scores"][c]["score"]) for cid, g in human.items()
                 if c in g and cid in by_id and c in by_id[cid]["scores"]]
        if not pairs:
            continue
        people, ai = [p for p, _ in pairs], [m for _, m in pairs]
        result[c] = {
            "calls": len(pairs),
            "exact": round(sum(p == m for p, m in pairs) / len(pairs), 2),
            "within_1": round(sum(abs(p - m) <= 1 for p, m in pairs) / len(pairs), 2),
            "mean_difference": round(statistics.mean(m - p for p, m in pairs), 2),   # + = AI more generous
            "kappa": weighted_kappa(people, ai),
        }
    return result


def kappa_words(k: float | None) -> str:
    if k is None:
        return "–"
    for limit, words in ((0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"), (0.2, "fair"), (0.0, "slight")):
        if k > limit:
            return words
    return "no better than chance"


def report(result: dict) -> str:
    out = ["## Does the AI reviewer agree with a person?", "",
           "| Criterion | Calls | Same grade | Within 1 point | AI − person (average) | Weighted kappa |",
           "|---|---|---|---|---|---|"]
    for c, r in result.items():
        out.append(f"| {c} | {r['calls']} | {r['exact']:.0%} | {r['within_1']:.0%} | {r['mean_difference']:+.2f} | "
                   f"{r['kappa'] if r['kappa'] is not None else '–'} ({kappa_words(r['kappa'])}) |")
    out += ["", "_AI − person: + means the AI reviewer gives higher grades than the person. "
                "Kappa: 1 = perfect agreement, 0 = no better than chance (quadratic weights)._", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", help="the batch to sample; all saved calls if not given")
    e.add_argument("--calls", type=int, default=10)
    sub.add_parser("compare")
    args = parser.parse_args()

    if args.step == "export":
        rows = [r for r in load_calls(run_id=args.run) if r["scores"]]
        if not rows:
            raise SystemExit("No reviewed calls saved. Run scripts/run_calls.py first.")
        if CSV_PATH.exists():
            raise SystemExit(f"{CSV_PATH} already exists: delete it first if you really want new calls.")
        csv_path, html_path = export(pick(rows, args.calls))
        print(f"1. Read the calls:  {html_path}\n2. Grade them in:   {csv_path}\n3. Then run:        "
              f"python scripts/judge_agreement.py compare")
    else:
        human = read_grades()
        if not human:
            raise SystemExit(f"No grades found in {CSV_PATH}: fill in the 4 columns (1 to 5) and save as CSV.")
        result = agreement(human, load_calls())
        text = report(result)
        print(text)
        REPORT_PATH.write_text(text, encoding="utf-8")
        print(f"Written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
