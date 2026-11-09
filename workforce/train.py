"""Trains the RL floor manager with PPO (Stable-Baselines3) on simulated days.

Run it from the project folder:
    python -m workforce.train                  # 400,000 steps = about 7,700 simulated days (~20 min)
    python -m workforce.train --seed 1         # a second agent, to check the result is stable

    python -m workforce.train --twin           # an agent for the twin's floor (after python -m workforce.twin)

The trained agents are saved in workforce/models/ (ppo_staffing.zip, ppo_staffing_seed1.zip, ...).
The twin agent is saved as ppo_twin.zip: it trains on the floor the AI calls imply (more calls, more
complaints, read from workforce/models/twin.json), with the same random surprise days on top.
Training days are random surprises (workforce/env.py, random_surprises).
Then compare it with the baseline:  python -m workforce.compare
"""

import argparse
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ppo_staffing.zip"
TRAIN_SEED = 0


def model_path(seed: int = TRAIN_SEED) -> Path:
    return MODEL_PATH if seed == TRAIN_SEED else MODEL_PATH.with_name(f"ppo_staffing_seed{seed}.zip")


def train(steps: int = 400_000, path: Path | None = None, seed: int = TRAIN_SEED, verbose: int = 1, reward=None,
          config=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    from workforce.env import CallCenterEnv

    path = path or model_path(seed)
    envs = make_vec_env(CallCenterEnv, n_envs=4, seed=seed, env_kwargs={"reward": reward, "config": config})  # 4 days side by side
    model = PPO(
        "MlpPolicy",
        envs,
        n_steps=52 * 4,            # 4 full days per environment between updates
        batch_size=208,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )
    model.learn(total_timesteps=steps, progress_bar=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL floor manager.")
    parser.add_argument("--steps", type=int, default=400_000, help="decisions to practice (52 per day)")
    parser.add_argument("--seed", type=int, default=TRAIN_SEED, help="train several agents with different seeds")
    parser.add_argument("--twin", action="store_true",
                        help="train on the twin's floor (the workload of workforce/models/twin.json)")
    args = parser.parse_args()
    if args.twin:
        from workforce.twin import TWIN_MODEL_PATH, saved_measured, twin_config
        measured = saved_measured()
        if measured is None:
            raise SystemExit("No twin measurement yet. Run first: python -m workforce.twin --run night-2")
        train(args.steps, path=TWIN_MODEL_PATH, seed=args.seed, config=twin_config(measured))
        print(f"Saved the twin agent to {TWIN_MODEL_PATH}")
    else:
        train(args.steps, seed=args.seed)
        print(f"Saved the agent to {model_path(args.seed)}")
