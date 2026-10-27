"""Run a batch of pretend calls and save every result for the dashboard.

Run it from the project folder, with Ollama running, the database filled and the rules indexed:

    python scripts/run_calls.py --calls 50 --run night-1          # about 1 to 2 hours on a laptop
    python scripts/run_calls.py --calls 50 --run night-1          # again: continues where it stopped

Each call takes 1 to 3 minutes (four AI workers talk). Calls are saved one by one in the
`calls` and `call_scores` tables, so you can stop with Ctrl+C at any time and start again
with the same --run: the calls already saved are skipped, and the next ones are the same
customers, reasons and moods as they would have been (same --seed).

A call that crashes is reported and skipped; the batch goes on.
Note: complaint calls open real tickets. scripts/seed_data.py resets the customers and tickets
but never deletes the saved call results.
"""

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.agents.client.scenarios import random_call  # noqa: E402
from callverse.graph import run_call  # noqa: E402
from callverse.results import load_calls, save_call, saved_call_ids, summary  # noqa: E402


def run(calls: int, run_id: str, seed: int, call=run_call, save=save_call, already=None, draw=random_call) -> dict:
    """Runs the batch. Returns {"done", "skipped", "failed"}. The workers are parameters for the tests."""
    rng = random.Random(seed)
    already = saved_call_ids(run_id) if already is None else already
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for number in range(1, calls + 1):
        new = draw(rng, f"{run_id}-{number:04d}")          # always drawn, so a restart gives the same calls
        if new.call_id in already:
            counts["skipped"] += 1
            continue
        start = time.time()
        try:
            done = call(new)
            save(done, run_id=run_id, seconds=time.time() - start)
        except KeyboardInterrupt:
            raise
        except Exception:
            counts["failed"] += 1
            print(f"[{number}/{calls}] {new.call_id} FAILED, skipped:\n{traceback.format_exc(limit=3)}")
            continue
        counts["done"] += 1
        quality = f"{done.quality_report.overall:.1f}" if done.quality_report else "?"
        print(f"[{number}/{calls}] {new.call_id}  {new.call_reason:<26} {new.persona:<20} -> "
              f"{done.outcome:<9} satisfaction {done.satisfaction}/5  quality {quality}/5  "
              f"({time.time() - start:.0f} s)", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--run", default="run-1", help="the name of this batch; reuse it to continue a batch")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    counts = run(args.calls, args.run, args.seed)
    print(f"\nDone: {counts['done']} new, {counts['skipped']} already saved, {counts['failed']} failed.")
    s = summary(load_calls(run_id=args.run))
    if s["calls"]:
        print(f"Run '{args.run}': {s['calls']} calls, resolved {s['resolved_rate']:.0%}, "
              f"satisfaction {s['avg_satisfaction']}/5, quality {s['avg_quality']}/5")


if __name__ == "__main__":
    main()
