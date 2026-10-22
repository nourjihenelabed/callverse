"""Checks the RL environment (workforce/env.py) and a very short training run.

Skipped on computers without the RL packages:  pip install -e ".[rl]"
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from workforce.env import MAX_CHANGE, CallCenterEnv, RLPolicy  # noqa: E402
from workforce.manager import FixedRulePolicy  # noqa: E402
from workforce.simulation import CallCenter  # noqa: E402


def test_environment_follows_the_gymnasium_rules():
    from gymnasium.utils.env_checker import check_env
    check_env(CallCenterEnv(), skip_render_check=True)


def test_one_episode_is_one_day_of_52_decisions():
    env = CallCenterEnv()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    steps, done = 0, False
    while not done:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward) and reward <= 0
        steps += 1
    assert steps == 52
    assert "kpis" in info


def test_no_correction_gives_exactly_the_baseline_day():
    env = CallCenterEnv()
    env.reset(options={"day": 42, "scenario": "normal"})
    done, keep_plan = False, np.array([MAX_CHANGE] * 3 + [0])
    while not done:
        _, _, done, _, info = env.step(keep_plan)
    assert info["kpis"] == CallCenter(FixedRulePolicy(), seed=42).run().kpis()


def test_short_training_produces_a_usable_agent(tmp_path):
    from workforce.train import train
    path = tmp_path / "agent.zip"
    train(steps=832, path=path, verbose=0)           # 4 environments x 208 steps: a smoke test, not a real training
    agent = RLPolicy.load(path)
    kpis = CallCenter(agent, seed=3).run().kpis()
    assert kpis["calls"] > 0 and 0 <= kpis["sla"] <= 1


def test_environment_can_be_set_to_one_scenario():
    env = CallCenterEnv(scenarios={"network_outage": 1.0})
    _, info = env.reset(seed=3)
    assert info["scenario"] == "network_outage"
    with pytest.raises(ValueError):
        CallCenterEnv(scenarios={"alien_invasion": 1.0})


def test_random_training_days_cover_normal_quiet_and_peaks():
    from workforce.env import random_surprises

    rng = np.random.default_rng(0)
    days = [random_surprises(rng) for _ in range(300)]
    assert any(d == () for d in days)                                        # normal days
    assert any(d and all(s.factor < 1 for s in d) for d in days)             # quieter than forecast
    assert any(d and any(s.factor > 1 for s in d) for d in days)             # surprise peaks
    for day in days:
        for surge in day:
            assert 8 <= surge.start_hour < surge.end_hour <= 20


def test_explanation_charts_are_built():
    pytest.importorskip("plotly")
    from workforce.explain import timeline
    from workforce.train import train

    import tempfile
    from pathlib import Path
    path = Path(tempfile.mkdtemp()) / "agent.zip"
    train(steps=832, path=path, verbose=0)
    fig = timeline(RLPolicy.load(path), "network_outage", day=1)
    assert len(fig.data) >= 12                                               # 4 policies x 3 queues, + bars
