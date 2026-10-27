"""Compares two batches of calls, with the statistics to know if a difference is real or chance.

Run it from the project folder:
    python scripts/compare_runs.py night-1 night-2
    python scripts/compare_runs.py night-1 night-2 --markdown evaluation.md     # tables for the report

It prints, for each batch:
- the headline numbers, and for each one whether the difference is significant (p < 0.05):
  a two-proportion z-test for shares (resolved), a Welch test for averages (satisfaction, grades);
- "quality without the outcome": the average of relevance, accuracy, compliance and empathy only,
  because resolution and satisfaction come straight from how the call ended;
- the results by customer mood, and the numbers of the retention route.

With 50 calls per batch, only big differences can be significant: say so in the report.
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.results import by, load_calls  # noqa: E402
from callverse.state import QUALITY_CRITERIA  # noqa: E402

HANDLED = ("resolved", "escalated")
WITHOUT_OUTCOME = ("relevance", "accuracy", "compliance", "empathy")


# ---------------------------------------------------------------- statistics (plain Python)

def _p_from_z(z: float) -> float:
    """Two-sided p-value of a normal score."""
    return math.erfc(abs(z) / math.sqrt(2))


def two_proportions(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float | None:
    """p-value of the difference between two shares (z-test with the pooled share)."""
    if not n_a or not n_b:
        return None
    pooled = (hits_a + hits_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return 1.0 if se == 0 else _p_from_z((hits_b / n_b - hits_a / n_a) / se)


def welch(a: list[float], b: list[float]) -> float | None:
    """p-value of the difference between two averages (Welch's test, normal approximation: fine from ~30 calls)."""
    if len(a) < 2 or len(b) < 2:
        return None
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return 1.0 if se == 0 else _p_from_z((statistics.mean(b) - statistics.mean(a)) / se)


def verdict(p: float | None) -> str:
    if p is None:
        return "–"
    return f"p = {p:.2f} " + ("(significant)" if p < 0.05 else "(could be chance)")


# ---------------------------------------------------------------- the comparison

def quality_without_outcome(row: dict) -> float | None:
    grades = [row["scores"].get(c, {}).get("score") for c in WITHOUT_OUTCOME]
    grades = [g for g in grades if g is not None]
    return statistics.mean(grades) if grades else None


def values(rows: list[dict], get) -> list[float]:
    return [v for v in (get(r) for r in rows) if v is not None]


def compare(a: list[dict], b: list[dict]) -> list[tuple[str, str, str, str]]:
    """Lines of (measure, batch A, batch B, test)."""
    def share(rows, test):
        return sum(1 for r in rows if test(r)), len(rows)

    lines = []
    for label, test in (("Resolved", lambda r: r["outcome"] == "resolved"),
                        ("Handled (resolved or escalated)", lambda r: r["outcome"] in HANDLED)):
        ha, na = share(a, test)
        hb, nb = share(b, test)
        lines.append((label, f"{ha}/{na} ({ha / na:.0%})" if na else "–", f"{hb}/{nb} ({hb / nb:.0%})" if nb else "–",
                      verdict(two_proportions(ha, na, hb, nb))))

    measures = [("Customer satisfaction", lambda r: r["satisfaction"]),
                ("Quality (all 6 criteria)", lambda r: r["overall"]),
                ("Quality without the outcome (4 criteria)", quality_without_outcome)]
    measures += [(f"  {c}", lambda r, c=c: r["scores"].get(c, {}).get("score")) for c in QUALITY_CRITERIA]
    for label, get in measures:
        va, vb = values(a, get), values(b, get)
        lines.append((label, f"{statistics.mean(va):.2f}" if va else "–", f"{statistics.mean(vb):.2f}" if vb else "–",
                      verdict(welch(va, vb))))
    return lines


def by_mood(a: list[dict], b: list[dict]) -> list[tuple[str, str, str]]:
    ga, gb = by(a, "persona"), by(b, "persona")
    lines = []
    for mood in sorted(set(ga) | set(gb)):
        def cell(groups):
            s = groups.get(mood)
            if not s:
                return "–"
            resolved = round((s["resolved_rate"] or 0) * s["calls"])
            return f"{resolved}/{s['calls']} resolved, satisfaction {s['avg_satisfaction']}"
        lines.append((mood, cell(ga), cell(gb)))
    return lines


def retention_lines(rows: list[dict]) -> str:
    leaving = [r for r in rows if r.get("wants_to_leave")]
    if not leaving:
        return "no customer flagged as wanting to leave (or the batch is older than the retention route)"
    offers = sum(1 for r in leaving if r.get("retention_offer"))
    handled = sum(1 for r in leaving if r["outcome"] in HANDLED)
    return (f"{len(leaving)} wanted to leave, {offers / len(leaving):.0%} got a supervisor's offer, "
            f"{handled / len(leaving):.0%} handled (resolved or with a supervisor)")


def markdown(name_a: str, name_b: str, a: list[dict], b: list[dict]) -> str:
    out = [f"## {name_a} vs {name_b}", "", f"| Measure | {name_a} | {name_b} | Test |", "|---|---|---|---|"]
    out += [f"| {m.strip()} | {x} | {y} | {t} |" for m, x, y, t in compare(a, b)]
    out += ["", "### By customer mood", "", f"| Mood | {name_a} | {name_b} |", "|---|---|---|"]
    out += [f"| {m} | {x} | {y} |" for m, x, y in by_mood(a, b)]
    out += ["", "### Retention route", "", f"- {name_a}: {retention_lines(a)}", f"- {name_b}: {retention_lines(b)}",
            "", f"_{len(a)} and {len(b)} calls. p < 0.05 means the difference is unlikely to be chance._", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", nargs="?", default="night-1")
    parser.add_argument("run_b", nargs="?", default="night-2")
    parser.add_argument("--markdown", type=Path, help="also write the tables to this file")
    args = parser.parse_args()

    a, b = load_calls(run_id=args.run_a), load_calls(run_id=args.run_b)
    if not a or not b:
        raise SystemExit(f"Need saved calls in both batches ({args.run_a}: {len(a)}, {args.run_b}: {len(b)}).")

    width = 42
    print(f"\n{'':<{width}}{args.run_a:>16}{args.run_b:>16}   test")
    for measure, x, y, test in compare(a, b):
        print(f"{measure:<{width}}{x:>16}{y:>16}   {test}")
    print("\nBy customer mood:")
    for mood, x, y in by_mood(a, b):
        print(f"  {mood:<20} {args.run_a}: {x:<36} {args.run_b}: {y}")
    print("\nRetention route:")
    print(f"  {args.run_a}: {retention_lines(a)}")
    print(f"  {args.run_b}: {retention_lines(b)}")
    if args.markdown:
        args.markdown.write_text(markdown(args.run_a, args.run_b, a, b), encoding="utf-8")
        print(f"\nTables written to {args.markdown}")


if __name__ == "__main__":
    main()
