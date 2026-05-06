"""Unit tests for ridge/rewards.py."""

import numpy as np
import pytest

from ridge.game import ACHIEVEMENTS
from ridge.rewards import (
    compute_blended_reward,
    craftsman_reward,
    explorer_reward,
    sigma,
    survivor_reward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_info() -> dict:
    return {
        "achievements": {name: 0 for name in ACHIEVEMENTS},
        "health": 9,
        "food": 9,
        "drink": 9,
        "energy": 9,
        "inventory": {
            "wood": 0, "stone": 0, "coal": 0, "iron": 0, "diamond": 0, "sapling": 0,
            "wood_pickaxe": 0, "stone_pickaxe": 0, "iron_pickaxe": 0,
            "wood_sword": 0, "stone_sword": 0, "iron_sword": 0,
        },
        "visited_count": 0,
        "steps_this_episode": 0,
        "player_pos": (5, 5),
    }


@pytest.fixture
def full_achievements_info(empty_info) -> dict:
    info = dict(empty_info)
    info["achievements"] = {name: 1 for name in ACHIEVEMENTS}
    return info


@pytest.fixture
def low_health_info(empty_info) -> dict:
    info = dict(empty_info)
    info["health"] = 1
    info["food"] = 1
    info["drink"] = 1
    return info


@pytest.fixture
def high_craft_info(empty_info) -> dict:
    info = dict(empty_info)
    info["inventory"] = dict(empty_info["inventory"])
    info["inventory"]["wood"] = 10
    info["inventory"]["iron"] = 5
    info["achievements"] = dict(empty_info["achievements"])
    info["achievements"]["make_iron_pickaxe"] = 1
    info["achievements"]["make_iron_sword"] = 1
    return info


# ---------------------------------------------------------------------------
# Persona reward functions
# ---------------------------------------------------------------------------

class TestExplorerReward:
    def test_zero_for_empty_info(self, empty_info):
        assert explorer_reward(empty_info) == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_achievements(self, full_achievements_info):
        assert explorer_reward(full_achievements_info) > 0.0

    def test_returns_float(self, empty_info):
        assert isinstance(explorer_reward(empty_info), float)


class TestSurvivorReward:
    def test_positive_at_full_health(self, empty_info):
        assert survivor_reward(empty_info) > 0.0

    def test_penalised_for_low_vitals(self, low_health_info, empty_info):
        assert survivor_reward(low_health_info) < survivor_reward(empty_info)

    def test_returns_float(self, empty_info):
        assert isinstance(survivor_reward(empty_info), float)


class TestCraftsmanReward:
    def test_zero_for_empty_info(self, empty_info):
        assert craftsman_reward(empty_info) == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_crafting(self, high_craft_info):
        assert craftsman_reward(high_craft_info) > 0.0

    def test_returns_float(self, empty_info):
        assert isinstance(craftsman_reward(empty_info), float)


# ---------------------------------------------------------------------------
# Sigma blending engine
# ---------------------------------------------------------------------------

class TestSigma:
    @pytest.fixture
    def base_config(self):
        return {
            "sigmoid_temperature": 1.0,
            "health_threshold": 0.3,
            "hunger_threshold": 0.3,
            "progress_threshold": 0.5,
        }

    def test_output_shape(self, base_config):
        sv = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        w = sigma(sv, base_config)
        assert w.shape == (3,)

    def test_weights_sum_to_one(self, base_config):
        sv = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        w = sigma(sv, base_config)
        assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self, base_config):
        for _ in range(20):
            sv = np.random.rand(6).astype(np.float32)
            w = sigma(sv, base_config)
            assert (w >= 0).all()

    def test_survivor_high_when_low_health(self, base_config):
        sv = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.0], dtype=np.float32)
        w = sigma(sv, base_config)
        assert w[1] > w[0], "Survivor weight should dominate when health/food/drink are critically low"

    def test_no_hard_switch(self, base_config):
        sv1 = np.array([0.31, 0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        sv2 = np.array([0.29, 0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        w1 = sigma(sv1, base_config)
        w2 = sigma(sv2, base_config)
        # No hard switch: weights should change smoothly, not jump by more than 0.5
        assert np.abs(w1 - w2).max() < 0.5


# ---------------------------------------------------------------------------
# Blending engine
# ---------------------------------------------------------------------------

class TestComputeBlendedReward:
    @pytest.fixture
    def state_vec(self):
        return np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.0], dtype=np.float32)

    @pytest.fixture
    def ridge_config(self):
        return {
            "blending_mode": "ridge",
            "sigmoid_temperature": 1.0,
            "health_threshold": 0.3,
            "hunger_threshold": 0.3,
            "progress_threshold": 0.5,
        }

    def test_returns_correct_types(self, empty_info, state_vec, ridge_config):
        reward, weights, per_persona = compute_blended_reward(empty_info, state_vec, ridge_config)
        assert isinstance(reward, float)
        assert weights.shape == (3,)
        assert set(per_persona.keys()) == {"explorer", "survivor", "craftsman"}

    def test_fixed_explorer_uses_only_explorer(self, empty_info, state_vec):
        config = {"blending_mode": "fixed_explorer"}
        _, weights, _ = compute_blended_reward(empty_info, state_vec, config)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(0.0)
        assert weights[2] == pytest.approx(0.0)

    def test_fixed_survivor(self, empty_info, state_vec):
        config = {"blending_mode": "fixed_survivor"}
        _, weights, _ = compute_blended_reward(empty_info, state_vec, config)
        assert weights[1] == pytest.approx(1.0)

    def test_fixed_craftsman(self, empty_info, state_vec):
        config = {"blending_mode": "fixed_craftsman"}
        _, weights, _ = compute_blended_reward(empty_info, state_vec, config)
        assert weights[2] == pytest.approx(1.0)

    def test_ridge_weights_sum_to_one(self, empty_info, state_vec, ridge_config):
        _, weights, _ = compute_blended_reward(empty_info, state_vec, ridge_config)
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)
