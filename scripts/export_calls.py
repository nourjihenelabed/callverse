"""Exports saved calls as one readable text file, to read why some calls fail.

Run it from the project folder:
    python scripts/export_calls.py night-2 night-3 night-3b              # complaints and customers leaving
    python scripts/export_calls.py night-3 --all                          # every call of a batch
    python scripts/export_calls.py night-2 night-3 --reason lost_sim      # one call reason

It writes evaluation/calls_<batches>.txt: for each call, the customer, reason and mood, how it ended,
the specialists' decisions, the whole conversation, and the reviewer's grades. The same customer and
reason appear in the same order in every batch (same seed), so the batches can be read side by side.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from callverse.results import load_calls  # noqa: E402


def interesting(row: dict) -> bool:
    """The calls where night-3 lost the most: complaints, and customers who want to leave."""
    return bool(row.get("is_complaint") or row.get("wants_to_leave")
                or str(row.get("call_reason", "")).startswith(("complaint", "cancel")))


def describe(row: dict) -> str:
    lines = [f"=== {row['call_id']}   customer {row['customer_id']}   {row['call_reason']}   mood {row['persona']}",
             f"    ended: {row['outcome']}   satisfaction {row['satisfaction']}/5   quality {row['overall']}   "
             f"turns {row['turns']}   intent {row['intent']}"]
    if row.get("complaint_category"):
        lines.append(f"    complaint specialist: {row['complaint_category']}, severity {row['complaint_severity']}, "
                     f"{(row['compensation_cents'] or 0) / 100:.2f} EUR, escalated={row['escalated']}")
    if row.get("wants_to_leave"):
        what = row.get("retention_offer") or ("manager calls back" if row.get("retention_supervisor") else "no offer")
        lines.append(f"    wants to leave: {what}")
    lines.append("")
    for m in row["messages"]:
        lines.append(f"  {'CLIENT ' if m['role'] == 'client' else 'ADVISOR'}: {m['content']}")
    lines.append("")
    for name, s in row["scores"].items():
        lines.append(f"    {name} {s['score']}/5: {s['explanation']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="the batches to export, e.g. night-2 night-3")
    parser.add_argument("--all", action="store_true", help="every call, not only complaints and leavers")
    parser.add_argument("--reason", help="only this call reason")
    args = parser.parse_args()

    blocks, counts = [], {}
    for run in args.runs:
        rows = [r for r in load_calls(run_id=run)
                if (args.all or interesting(r)) and (not args.reason or r["call_reason"] == args.reason)]
        counts[run] = len(rows)
        blocks.append(f"\n\n################ {run}: {len(rows)} calls ################\n")
        blocks += [describe(r) for r in sorted(rows, key=lambda r: r["call_id"])]
    out = ROOT / "evaluation" / f"calls_{'_'.join(args.runs)}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(", ".join(f"{run}: {n} calls" for run, n in counts.items()) + f"\nWritten to {out}")


if __name__ == "__main__":
    main()
