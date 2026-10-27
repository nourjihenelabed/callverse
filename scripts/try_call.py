"""Run whole pretend calls: the AI customer talks to the AI advisor, start to finish.

Run it from the project folder, with Ollama running, the database filled
(python scripts/seed_data.py) and the rules indexed (python scripts/build_index.py):

    python scripts/try_call.py                                   # 3 random calls
    python scripts/try_call.py --calls 5 --seed 7
    python scripts/try_call.py --reason complaint_billing --persona demanding
    python scripts/try_call.py --customer 7590-VHVEG --reason lost_sim

It prints each conversation, the outcome, the customer's satisfaction and the reviewer's report
(a grade from 1 to 5 for each of the 6 criteria, with the reason). Add --short to hide the reasons.
A call takes about 1 to 3 minutes on a laptop (several AI answers per turn).
Note: complaint calls open real tickets in your database. Run seed_data.py again to clean them.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.agents.client.personas import PERSONAS  # noqa: E402
from callverse.agents.client.scenarios import REASONS, random_call  # noqa: E402
from callverse.graph import run_call  # noqa: E402
from callverse.state import CallState  # noqa: E402

LABEL = {"client": "CLIENT  ", "advisor": "ADVISOR "}


def show_review(done: CallState, short: bool) -> None:
    r = done.quality_report
    if r is None:
        return
    escalation = "" if r.escalation_correct is None else f"   escalation correct: {r.escalation_correct}"
    print(f"  REVIEW: overall {r.overall:.1f}/5   "
          + "  ".join(f"{name} {s.score}" for name, s in r.scores.items()) + escalation)
    if not short:
        for name, s in r.scores.items():
            print(f"    {name} {s.score}/5: {s.explanation}")
        for tip in r.recommendations:
            print(f"    tip: {tip}")


def show(done: CallState, seconds: float) -> None:
    for message in done.messages:
        print(f"  {LABEL[message.role]}: {message.content}")
    if done.complaint_result:
        r = done.complaint_result
        print(f"  [complaint specialist: {r.category}, severity {r.severity}, "
              f"{r.compensation_cents / 100:.2f} EUR, escalate={r.escalate}, ticket {r.ticket_id}]")
    if done.retention:
        r = done.retention
        what = r.offer or ("supervisor calls back" if r.supervisor else "conditions explained, no offer")
        print(f"  [supervisor: customer wants to leave, high risk={r.high_risk}: {what}]")
    tools = ", ".join(sorted({t.name for t in done.tools_used})) or "none"
    print(f"  -> outcome: {done.outcome}   satisfaction: {done.satisfaction}/5   turns: {done.turn}   "
          f"tools: {tools}   ({seconds:.0f} s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--customer", help="a customer id; random if not given")
    parser.add_argument("--reason", choices=sorted(REASONS), help="random if not given")
    parser.add_argument("--persona", choices=sorted(PERSONAS), help="random if not given")
    parser.add_argument("--short", action="store_true", help="grades only, without the reasons")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    outcomes = []
    for number in range(1, args.calls + 1):
        call = random_call(rng, f"try-{args.seed}-{number}", customer_id=args.customer)
        changes = {k: v for k, v in (("call_reason", args.reason), ("persona", args.persona)) if v}
        call = call.model_copy(update=changes)
        print(f"\n=== Call {number}: customer {call.customer_id}, reason {call.call_reason}, mood {call.persona}")
        start = time.time()
        done = run_call(call)
        show(done, time.time() - start)
        show_review(done, args.short)
        outcomes.append((done.outcome, done.satisfaction, done.quality_report.overall if done.quality_report else None))

    if len(outcomes) > 1:
        satisfaction = [s for _, s, _ in outcomes if s]
        quality = [q for _, _, q in outcomes if q]
        print(f"\n{len(outcomes)} calls: " + ", ".join(o or "?" for o, _, _ in outcomes)
              + f"   average satisfaction {sum(satisfaction) / max(1, len(satisfaction)):.1f}/5"
              + f"   average quality {sum(quality) / max(1, len(quality)):.1f}/5")


if __name__ == "__main__":
    main()
