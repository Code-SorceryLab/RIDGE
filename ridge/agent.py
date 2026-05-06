import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

N_PERSONAS = 4  # explorer | survivor | craftsman | warrior


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Shared convolutional encoder for the 64×64 RGB Crafter observation."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observation batch.

        Args:
            x: Float tensor of shape (B, 3, 64, 64), values in [0, 1].

        Returns:
            Feature tensor of shape (B, 512).
        """
        return self.net(x)  # (B, 512)


class PolicyHead(nn.Module):
    """Maps encoder features to action logits."""

    def __init__(self, feature_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            features: Tensor of shape (B, feature_dim).

        Returns:
            Logits tensor of shape (B, num_actions).
        """
        return self.net(features)  # (B, num_actions)


class ValueHead(nn.Module):
    """Single value head — one instantiated per persona."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate state value.

        Args:
            features: Tensor of shape (B, feature_dim).

        Returns:
            Value tensor of shape (B, 1).
        """
        return self.net(features)  # (B, 1)


class RIDGENetwork(nn.Module):
    """Full network: shared encoder + policy head + 3 value heads."""
    """Shared CNN encoder + policy head + 4 independent value heads.

    Head order: [explorer, survivor, craftsman, warrior]
    This order must match PERSONA_NAMES in rewards.py.
    """

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.encoder = CNNEncoder()
        self.policy_head = PolicyHead(self.encoder.output_dim, num_actions)
        self.value_head_explorer = ValueHead(self.encoder.output_dim)
        self.value_head_survivor = ValueHead(self.encoder.output_dim)
        self.value_head_craftsman = ValueHead(self.encoder.output_dim)
        self.value_head_warrior   = ValueHead(self.encoder.output_dim)

    def forward(
        self, obs: torch.Tensor, persona_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            obs: Float tensor of shape (B, 3, 64, 64).
            persona_weights: Float tensor of shape (B, 3) or (3,) — [w_e, w_s, w_c].

        Returns:
            Tuple of:
              - logits: (B, num_actions)
              - value: (B, 1) — weighted sum of per-head values
              - per_head_values: (B, 3) — [V_explorer, V_survivor, V_craftsman]
        """
        features = self.encoder(obs)  # (B, 512)
        logits = self.policy_head(features)  # (B, num_actions)

        v_e = self.value_head_explorer(features)   # (B, 1)
        v_s = self.value_head_survivor(features)   # (B, 1)
        v_c = self.value_head_craftsman(features)  # (B, 1)
        v_w = self.value_head_warrior  (features)             # (B, 1)

        per_head = torch.cat([v_e, v_s, v_c, v_w], dim=-1)   # (B, 4)

        # Broadcast weights: (B, 3) or expand scalar batch
        if persona_weights.dim() == 1:
            w = persona_weights.unsqueeze(0).expand(features.shape[0], -1)  # (B, 4)
        else:
            w = persona_weights  # (B, 3)

        value = (per_head * w).sum(dim=-1, keepdim=True)      # (B, 1)
        return logits, value, per_head


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores one rollout's worth of transitions for PPO update."""

    obs:                  list[np.ndarray] = field(default_factory=list)
    actions:              list[int]        = field(default_factory=list)
    log_probs:            list[float]      = field(default_factory=list)
    values:               list[float]      = field(default_factory=list)
    per_head_values:      list[np.ndarray] = field(default_factory=list)  # (4,) V_i — bootstrap
    persona_step_rewards: list[np.ndarray] = field(default_factory=list)  # (4,) r_i — G_i targets
    rewards:              list[float]      = field(default_factory=list)
    dones:                list[bool]       = field(default_factory=list)
    persona_weights:      list[np.ndarray] = field(default_factory=list)  # (4,) blend weights
    infos:                list[dict]       = field(default_factory=list)

    def clear(self) -> None:
        """Reset buffer to empty."""
        for attr in ("obs", "actions", "log_probs", "values", "per_head_values",
                     "persona_step_rewards", "rewards", "dones", "persona_weights", "infos"):
            setattr(self, attr, [])

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent with RIDGE multi-head critic.

    Implements action selection, PPO updates with clipped surrogate loss,
    per-head value losses, and checkpoint save/load.
    """

    def __init__(self, config: dict[str, Any], num_actions: int, device: torch.device) -> None:
        """Initialise networks, optimiser, and hyperparameters.

        Args:
            config: Project config dict.
            num_actions: Discrete action space size from Crafter.
            device: Torch device (CPU or CUDA).
        """
        self._config       = config
        self._device       = device
        self._num_actions  = num_actions

        self.network   = RIDGENetwork(num_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=float(config["lr"]))

        self._gamma           = float(config.get("gamma",           0.99))
        self._gae_lambda      = float(config.get("gae_lambda",      0.95))
        self._clip_epsilon    = float(config.get("clip_epsilon",    0.2))
        self._entropy_coef    = float(config.get("entropy_coef",    0.01))
        self._value_coef      = float(config.get("value_coef",      0.5))
        self._max_grad_norm   = float(config.get("max_grad_norm",   0.5))
        self._ppo_epochs      = int  (config.get("ppo_epochs",      4))
        self._num_minibatches = int  (config.get("num_minibatches", 4))

        self.training_step = 0
        logger.info("PPOAgent initialised — %d actions on %s", num_actions, device)

    def select_action(
        self, obs: np.ndarray, persona_weights: np.ndarray
    ) -> tuple[int, float, float, np.ndarray]:
        """Sample an action from the policy.

        Args:
            obs: Float32 ndarray (3, 64, 64).
            persona_weights: Float32 ndarray (4,).

        Returns:
            Tuple of (action, log_prob, value, per_head_values).
            - action: Integer action index.
            - log_prob: Log probability of the selected action.
            - value: Blended value estimate (scalar float).
            - per_head_values: ndarray of shape (3,) — per-persona value estimates.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)  # (1, 3, 64, 64)
        w_t   = torch.as_tensor(persona_weights, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            logits, value, per_head = self.network(obs_t, w_t)

        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.squeeze().item()),
            per_head.squeeze(0).cpu().numpy(),  # (4,)
        )

    def compute_advantages(
        self,
        buffer: RolloutBuffer,
        last_value: float,
        last_per_head_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute blended GAE advantages and per-persona returns.

        Returns:
            advantages (T,), returns (T,), per_persona_returns (T, 4)
        """
        T = len(buffer)
        advantages          = np.zeros(T, dtype=np.float32)
        returns             = np.zeros(T, dtype=np.float32)
        per_persona_returns = np.zeros((T, N_PERSONAS), dtype=np.float32)

        persona_rewards_arr = np.stack(buffer.persona_step_rewards)  # (T, 4)
        per_head_arr        = np.stack(buffer.per_head_values)        # (T, 4)

        # Blended GAE — drives the actor
        gae = 0.0
        for t in reversed(range(T)):
            next_v   = last_value if t == T - 1 else buffer.values[t + 1]
            nont     = 1.0 - float(buffer.dones[t])
            delta    = buffer.rewards[t] + self._gamma * next_v * nont - buffer.values[t]
            gae      = delta + self._gamma * self._gae_lambda * nont * gae
            advantages[t] = gae
            returns[t]    = gae + buffer.values[t]

        # Per-persona TD(λ) returns — independent bootstrap per head
        gae_per = np.zeros(N_PERSONAS, dtype=np.float32)
        for t in reversed(range(T)):
            nont       = 1.0 - float(buffer.dones[t])
            next_v_per = last_per_head_value if t == T - 1 else per_head_arr[t + 1]
            delta_per  = (persona_rewards_arr[t]
                          + self._gamma * next_v_per * nont
                          - per_head_arr[t])
            gae_per    = delta_per + self._gamma * self._gae_lambda * nont * gae_per
            per_persona_returns[t] = gae_per + per_head_arr[t]

        return advantages, returns, per_persona_returns

    def ppo_update(
        self,
        buffer: RolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
        per_persona_returns: np.ndarray,
    ) -> dict[str, float]:
        """Run PPO epochs over the rollout buffer and return training metrics.

        Args:
            buffer: Rollout buffer with T transitions.
            advantages: Float32 ndarray of shape (T,).
            returns: Float32 ndarray of shape (T,).

        Returns:
            Dict with keys: policy_loss, value_loss, entropy, kl_divergence,
            clip_fraction, value_loss_explorer, value_loss_survivor, value_loss_craftsman.
        """
        T          = len(buffer)
        batch_size = T // self._num_minibatches

        obs_t       = torch.as_tensor(np.stack(buffer.obs), dtype=torch.float32, device=self._device)
        actions_t   = torch.as_tensor(buffer.actions, dtype=torch.long,    device=self._device)
        old_lp_t    = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self._device)
        adv_t       = torch.as_tensor(advantages, dtype=torch.float32,      device=self._device)
        ret_t       = torch.as_tensor(returns,    dtype=torch.float32,      device=self._device)
        weights_t   = torch.as_tensor(np.stack(buffer.persona_weights), dtype=torch.float32, device=self._device)  # (T, 4)
        ph_tgt_t    = torch.as_tensor(per_persona_returns, dtype=torch.float32, device=self._device)               # (T, 4)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        metrics: dict[str, list[float]] = {k: [] for k in (
            "policy_loss", "value_loss", "entropy", "kl_divergence", "clip_fraction",
            "value_loss_explorer", "value_loss_survivor",
            "value_loss_craftsman", "value_loss_warrior",
        )}

        indices = np.arange(T)
        for _ in range(self._ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, batch_size):
                idx = indices[start: start + batch_size]
                if not len(idx):
                    continue

                logits, value, per_head = self.network(obs_t[idx], weights_t[idx])
                dist      = Categorical(logits=logits)
                log_probs = dist.log_prob(actions_t[idx])
                entropy   = dist.entropy().mean()

                ratio  = torch.exp(log_probs - old_lp_t[idx])
                surr1  = ratio * adv_t[idx]
                surr2  = torch.clamp(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(value.squeeze(-1), ret_t[idx])

                mb_tgt = ph_tgt_t[idx]  # (B, 4)
                vl_e = nn.functional.mse_loss(per_head[:, 0], mb_tgt[:, 0])
                vl_s = nn.functional.mse_loss(per_head[:, 1], mb_tgt[:, 1])
                vl_c = nn.functional.mse_loss(per_head[:, 2], mb_tgt[:, 2])
                vl_w = nn.functional.mse_loss(per_head[:, 3], mb_tgt[:, 3])
                per_head_loss = (vl_e + vl_s + vl_c + vl_w) / N_PERSONAS

                total_loss = (
                    policy_loss
                    + self._value_coef * (value_loss + per_head_loss)
                    - self._entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self._max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self._clip_epsilon).float().mean().item()
                    kl        = (old_lp_t[idx] - log_probs).mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["kl_divergence"].append(kl)
                metrics["clip_fraction"].append(clip_frac)
                metrics["value_loss_explorer"].append(vl_e.item())
                metrics["value_loss_survivor"].append(vl_s.item())
                metrics["value_loss_craftsman"].append(vl_c.item())
                metrics["value_loss_warrior"].append(vl_w.item())

        self.training_step += 1
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save_checkpoint(self, path: str) -> None:
        """Save full agent state to disk.

        Args:
            path: File path for the checkpoint (.pt file).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network_state_dict":   self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step":        self.training_step,
            "config":               self._config,
        }, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self._device)
        self.network.load_state_dict(ckpt["network_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step = ckpt.get("training_step", 0)
        logger.info("Checkpoint loaded from %s (step %d)", path, self.training_step)