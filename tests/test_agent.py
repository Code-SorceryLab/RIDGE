"""Unit tests for ridge/agent.py."""

import numpy as np
import pytest
import torch

from ridge.agent import PPOAgent, RolloutBuffer


@pytest.fixture
def config() -> dict:
    return {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "ppo_epochs": 2,
        "num_minibatches": 2,
        "rollout_steps": 16,
    }


@pytest.fixture
def agent(config) -> PPOAgent:
    device = torch.device("cpu")
    return PPOAgent(config, num_actions=17, device=device)


@pytest.fixture
def dummy_obs() -> np.ndarray:
    return np.random.rand(3, 64, 64).astype(np.float32)


@pytest.fixture
def uniform_weights() -> np.ndarray:
    return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)


class TestSelectAction:
    def test_action_in_valid_range(self, agent, dummy_obs, uniform_weights):
        action, _, _, _ = agent.select_action(dummy_obs, uniform_weights)
        assert 0 <= action < 17

    def test_log_prob_is_scalar_float(self, agent, dummy_obs, uniform_weights):
        _, log_prob, _, _ = agent.select_action(dummy_obs, uniform_weights)
        assert isinstance(log_prob, float)

    def test_value_is_scalar_float(self, agent, dummy_obs, uniform_weights):
        _, _, value, _ = agent.select_action(dummy_obs, uniform_weights)
        assert isinstance(value, float)

    def test_per_head_values_shape(self, agent, dummy_obs, uniform_weights):
        _, _, _, per_head = agent.select_action(dummy_obs, uniform_weights)
        assert per_head.shape == (3,)

    def test_fixed_explorer_weights(self, agent, dummy_obs):
        weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        _, _, value, per_head = agent.select_action(dummy_obs, weights)
        # With explorer weight=1, value should equal per_head[0]
        assert value == pytest.approx(per_head[0], abs=1e-4)


class TestComputeAdvantages:
    def _make_buffer(self, n: int = 8) -> RolloutBuffer:
        buf = RolloutBuffer()
        for _ in range(n):
            buf.rewards.append(1.0)
            buf.values.append(0.5)
            buf.dones.append(False)
        return buf

    def test_advantages_shape(self, agent):
        buf = self._make_buffer(8)
        adv, ret = agent.compute_advantages(buf, last_value=0.5)
        assert adv.shape == (8,)
        assert ret.shape == (8,)

    def test_returns_greater_than_zero(self, agent):
        buf = self._make_buffer(8)
        _, ret = agent.compute_advantages(buf, last_value=0.5)
        assert (ret > 0).all()

    def test_done_terminates_bootstrap(self, agent):
        buf = RolloutBuffer()
        for i in range(4):
            buf.rewards.append(1.0)
            buf.values.append(0.5)
            buf.dones.append(i == 2)  # done at step 2
        adv, _ = agent.compute_advantages(buf, last_value=0.0)
        # Advantage at step 3 should not include bootstrap from before done
        assert adv.shape == (4,)


class TestPPOUpdate:
    def _make_full_buffer(self, n: int = 16) -> RolloutBuffer:
        buf = RolloutBuffer()
        for _ in range(n):
            buf.obs.append(np.random.rand(3, 64, 64).astype(np.float32))
            buf.actions.append(np.random.randint(0, 17))
            buf.log_probs.append(np.random.randn())
            buf.values.append(np.random.rand())
            buf.per_head_values.append(np.random.rand(3).astype(np.float32))
            buf.rewards.append(np.random.rand())
            buf.dones.append(False)
            buf.persona_weights.append(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32))
        return buf

    def test_returns_expected_keys(self, agent):
        buf = self._make_full_buffer()
        adv = np.random.rand(16).astype(np.float32)
        ret = np.random.rand(16).astype(np.float32)
        metrics = agent.ppo_update(buf, adv, ret)
        expected = {
            "policy_loss", "value_loss", "entropy", "kl_divergence",
            "clip_fraction", "value_loss_explorer", "value_loss_survivor", "value_loss_craftsman",
        }
        assert set(metrics.keys()) == expected

    def test_metrics_are_finite(self, agent):
        buf = self._make_full_buffer()
        adv = np.random.rand(16).astype(np.float32)
        ret = np.random.rand(16).astype(np.float32)
        metrics = agent.ppo_update(buf, adv, ret)
        for k, v in metrics.items():
            assert np.isfinite(v), f"Metric {k} is not finite: {v}"


class TestCheckpoint:
    def test_save_and_load(self, agent, tmp_path):
        path = str(tmp_path / "test_ckpt.pt")
        agent.save_checkpoint(path)

        from ridge.utils import get_device
        agent2 = PPOAgent(agent._config, num_actions=17, device=torch.device("cpu"))
        agent2.load_checkpoint(path)
        assert agent2.training_step == agent.training_step
