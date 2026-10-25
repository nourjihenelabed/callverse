"""Rebuilds every workforce result in one go, in the right order.

    python scripts/run_workforce.py                      # everything (about 1 hour 15 min on a laptop)
    python scripts/run_workforce.py --skip-calibration   # no Technion files: keep calibration.json as it is
    python scripts/run_workforce.py --steps 50000        # quick trial run (much weaker agents)

Steps:
1. calibrate from the Technion files (data/raw/technion/), roster included
2. tune the reactive and margin rules for this calibrated center
3. train the 3 RL agents (seeds 0, 1, 2)
4. compare all policies on the 8 test scenarios -> workforce/models/comparison.json + results.md
5. draw the explanation charts -> workforce/models/explain/
Then paste workforce/models/results.md into the "Results" section of workforce/README.md.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(title: str, command: list[str]) -> None:
    print(f"\n=== {title}", flush=True)
    start = time.time()
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    print(f"--- done in {(time.time() - start) / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild every workforce result.")
    parser.add_argument("--skip-calibration", action="store_true", help="keep workforce/calibration.json as it is")
    parser.add_argument("--steps", type=int, default=None, help="training steps per agent (default: train.py's)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not args.skip_calibration:
        step("1/5 Calibrate from the Technion files", ["scripts/calibrate_workforce.py"])
    step("2/5 Tune the reactive and margin rules", ["-m", "workforce.tune_rules"])
    for seed in args.seeds:
        extra = ["--steps", str(args.steps)] if args.steps else []
        step(f"3/5 Train RL agent, seed {seed}", ["-m", "workforce.train", "--seed", str(seed), *extra])
    step("4/5 Compare all policies on the 8 test scenarios", ["-m", "workforce.compare"])
    step("5/5 Draw the explanation charts", ["-m", "workforce.explain"])
    print("\nAll done. Paste workforce/models/results.md into the Results section of workforce/README.md.")


if __name__ == "__main__":
    main()
