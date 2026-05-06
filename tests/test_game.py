"""Unit tests for ridge/game.py — requires crafter to be installed."""

import numpy as np
import pytest

from ridge.game import ACHIEVEMENTS, CrafterWrapper, EpisodeStats, make_env


@pytest.fixture
def config() -> dict:
    return {
        "env_name": "Crafter",
        "frame_size": 64,
        "frame_stack": 1,
    }


@pytest.fixture
def env(config) -> CrafterWrapper:
    e = make_env(config, seed=0)
    yield e
    e.close()


class TestMakeEnv:
    def test_returns_crafter_wrapper(self, config):
        e = make_env(config, seed=42)
        assert isinstance(e, CrafterWrapper)
        e.close()

    def test_has_action_space(self, env):
        assert hasattr(env.action_space, "n")
        assert env.action_space.n > 0


class TestReset:
    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (3, 64, 64), f"Unexpected obs shape: {obs.shape}"

    def test_obs_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_range(self, env):
        obs, _ = env.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_info_has_required_keys(self, env):
        _, info = env.reset()
        for key in ("health", "food", "drink", "energy", "achievements", "inventory"):
            assert key in info, f"Missing key: {key}"


class TestStep:
    def test_step_returns_five_values(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (3, 64, 64)

    def test_step_reward_is_float(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_terminated_is_bool(self, env):
        env.reset()
        _, _, terminated, truncated, _ = env.step(0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_multiple_steps_dont_crash(self, env):
        env.reset()
        obs, info = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, done, _, info = env.step(action)
            if done:
                obs, info = env.reset()


class TestExtractStateVector:
    def test_output_shape(self, env):
        _, info = env.reset()
        sv = env.extract_state_vector(info)
        assert sv.shape == (6,)

    def test_output_dtype(self, env):
        _, info = env.reset()
        sv = env.extract_state_vector(info)
        assert sv.dtype == np.float32

    def test_values_in_zero_one(self, env):
        _, info = env.reset()
        sv = env.extract_state_vector(info)
        assert sv.min() >= 0.0
        assert sv.max() <= 1.0


class TestEpisodeStats:
    def test_update_increments_steps(self):
        stats = EpisodeStats()
        info = {"achievements": {name: 0 for name in ACHIEVEMENTS}}
        weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        stats.update(1.0, info, weights)
        assert stats.steps == 1

    def test_to_dict_keys(self):
        stats = EpisodeStats()
        d = stats.to_dict()
        for key in ("total_reward", "steps", "achievements_unlocked", "achievement_count", "mean_weights"):
            assert key in d

    def test_achievement_unlocked_tracked(self):
        stats = EpisodeStats()
        info = {"achievements": {name: 0 for name in ACHIEVEMENTS}}
        info["achievements"]["collect_wood"] = 1
        weights = np.zeros(3, dtype=np.float32)
        stats.update(0.0, info, weights)
        assert "collect_wood" in stats.achievements_unlocked
