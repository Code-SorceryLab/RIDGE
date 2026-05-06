import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

N_PERSONAS       = 4   # explorer | survivor | craftsman | warrior
N_ACHIEVEMENTS   = 22  # total Crafter achievements
ACH_HISTORY_LEN  = 5   # rolling window of recent unlocks
ACH_EMBED_DIM    = 16  # embedding dim per achievement → 5*16 = 80 extra features
PAD_IDX          = 0   # index used when history has fewer than 5 entries

# Canonical achievement index map — must match game.py ACHIEVEMENTS list order
ACHIEVEMENT_NAMES = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]
ACH_TO_IDX = {name: i + 1 for i, name in enumerate(ACHIEVEMENT_NAMES)}  # 1-indexed, 0=pad


# ---------------------------------------------------------------------------
# Achievement History Embedder
# ---------------------------------------------------------------------------

class AchievementHistoryEmbedder(nn.Module):
    """Embeds a fixed-length history of recently unlocked achievements.

    Takes the last ACH_HISTORY_LEN achievement indices (padded with 0 if fewer
    have been unlocked) and returns a flat embedding vector.

    This gives the policy a compact representation of where the agent is in
    the tech tree without any auxiliary loss — the embedding is learned purely
    through the policy gradient signal.

    Vocabulary: 0 = padding, 1–22 = achievement index (1-indexed)
    Output: (B, ACH_HISTORY_LEN * ACH_EMBED_DIM) = (B, 80)
    """

    def __init__(
        self,
        n_achievements: int = N_ACHIEVEMENTS,
        history_len: int    = ACH_HISTORY_LEN,
        embed_dim: int      = ACH_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.embed_dim   = embed_dim
        # +1 for padding token at index 0
        self.embedding = nn.Embedding(n_achievements + 1, embed_dim, padding_idx=PAD_IDX)
        self.output_dim = history_len * embed_dim  # 80

    def forward(self, history_indices: torch.Tensor) -> torch.Tensor:
        """Embed achievement history.

        Args:
            history_indices: LongTensor of shape (B, history_len),
                values in [0, N_ACHIEVEMENTS]. 0 = padding.

        Returns:
            Float tensor of shape (B, history_len * embed_dim).
        """
        embedded = self.embedding(history_indices)       # (B, 5, 16)
        return embedded.view(embedded.shape[0], -1)      # (B, 80)


def history_to_tensor(
    history: list[str],
    device: torch.device,
    history_len: int = ACH_HISTORY_LEN,
) -> torch.Tensor:
    """Convert a list of achievement name strings to a padded index tensor.

    Args:
        history: List of up to history_len achievement names (most recent last).
        device: Target torch device.
        history_len: Fixed window size (pad left with zeros if shorter).

    Returns:
        LongTensor of shape (1, history_len).
    """
    indices = [ACH_TO_IDX.get(name, PAD_IDX) for name in history[-history_len:]]
    # Left-pad with zeros to fixed length
    padded = [PAD_IDX] * (history_len - len(indices)) + indices
    return torch.tensor([padded], dtype=torch.long, device=device)  # (1, 5)


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
        return self.net(x)  # (B, 512)


class PolicyHead(nn.Module):
    def __init__(self, feature_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RIDGENetwork(nn.Module):
    """CNN encoder + achievement history embedder + policy head + 4 value heads.

    The CNN encodes the visual observation (512-dim).
    The history embedder encodes the last 5 achievements unlocked (80-dim).
    These are concatenated (592-dim) before the policy and value heads.

    This gives the policy implicit knowledge of where it is in the tech tree
    without any auxiliary loss — learned purely via policy gradient.
    """

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.encoder  = CNNEncoder()                     # → 512
        self.ach_emb  = AchievementHistoryEmbedder()     # → 80
        combined_dim  = self.encoder.output_dim + self.ach_emb.output_dim  # 592

        self.policy_head          = PolicyHead(combined_dim, num_actions)
        self.value_head_explorer  = ValueHead(combined_dim)
        self.value_head_survivor  = ValueHead(combined_dim)
        self.value_head_craftsman = ValueHead(combined_dim)
        self.value_head_warrior   = ValueHead(combined_dim)

    def forward(
        self,
        obs: torch.Tensor,
        persona_weights: torch.Tensor,
        ach_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            obs: (B, 3, 64, 64)
            persona_weights: (B, 4) or (4,)
            ach_history: (B, 5) LongTensor of achievement indices, 0=padding

        Returns:
            logits (B, num_actions), value (B, 1), per_head (B, 4)
        """
        vis_features = self.encoder(obs)                         # (B, 512)
        ach_features = self.ach_emb(ach_history)                 # (B, 80)
        features     = torch.cat([vis_features, ach_features], dim=-1)  # (B, 592)

        logits = self.policy_head(features)                      # (B, A)

        v_e = self.value_head_explorer (features)                # (B, 1)
        v_s = self.value_head_survivor (features)                # (B, 1)
        v_c = self.value_head_craftsman(features)                # (B, 1)
        v_w = self.value_head_warrior  (features)                # (B, 1)

        per_head = torch.cat([v_e, v_s, v_c, v_w], dim=-1)      # (B, 4)

        if persona_weights.dim() == 1:
            w = persona_weights.unsqueeze(0).expand(features.shape[0], -1)
        else:
            w = persona_weights

        value = (per_head * w).sum(dim=-1, keepdim=True)         # (B, 1)
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
    per_head_values:      list[np.ndarray] = field(default_factory=list)
    persona_step_rewards: list[np.ndarray] = field(default_factory=list)
    rewards:              list[float]      = field(default_factory=list)
    dones:                list[bool]       = field(default_factory=list)
    persona_weights:      list[np.ndarray] = field(default_factory=list)
    ach_histories:        list[np.ndarray] = field(default_factory=list)  # (5,) int — NEW
    infos:                list[dict]       = field(default_factory=list)

    def clear(self) -> None:
        for attr in ("obs", "actions", "log_probs", "values", "per_head_values",
                     "persona_step_rewards", "rewards", "dones", "persona_weights",
                     "ach_histories", "infos"):
            setattr(self, attr, [])

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent with RIDGE 4-head critic and achievement history embedding."""

    def __init__(self, config: dict[str, Any], num_actions: int, device: torch.device) -> None:
        self._config      = config
        self._device      = device
        self._num_actions = num_actions

        self.network   = RIDGENetwork(num_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=float(config["lr"]))

        self._gamma           = float(config.get("gamma",           0.99))
        self._gae_lambda      = float(config.get("gae_lambda",      0.95))
        self._clip_epsilon    = float(config.get("clip_epsilon",    0.2))
        self._entropy_coef    = float(config.get("entropy_coef",    0.01))
        self._value_coef      = float(config.get("value_coef",      0.1))
        self._max_grad_norm   = float(config.get("max_grad_norm",   0.5))
        self._ppo_epochs      = int  (config.get("ppo_epochs",      4))
        self._num_minibatches = int  (config.get("num_minibatches", 8))

        self.training_step = 0
        logger.info("PPOAgent initialised — %d actions on %s", num_actions, device)

    def select_action(
        self,
        obs: np.ndarray,
        persona_weights: np.ndarray,
        ach_history: list[str],
    ) -> tuple[int, float, float, np.ndarray]:
        """Sample an action from the policy.

        Args:
            obs: Float32 ndarray (3, 64, 64).
            persona_weights: Float32 ndarray (4,).
            ach_history: List of up to 5 recently unlocked achievement names.

        Returns:
            (action, log_prob, blended_value, per_head_values)
        """
        obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        w_t    = torch.as_tensor(persona_weights, dtype=torch.float32, device=self._device)
        hist_t = history_to_tensor(ach_history, self._device)  # (1, 5)

        with torch.no_grad():
            logits, value, per_head = self.network(obs_t, w_t, hist_t)

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
        """Compute blended GAE and per-persona returns."""
        T = len(buffer)
        advantages          = np.zeros(T, dtype=np.float32)
        returns             = np.zeros(T, dtype=np.float32)
        per_persona_returns = np.zeros((T, N_PERSONAS), dtype=np.float32)

        persona_rewards_arr = np.stack(buffer.persona_step_rewards)
        per_head_arr        = np.stack(buffer.per_head_values)

        gae = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else buffer.values[t + 1]
            nont   = 1.0 - float(buffer.dones[t])
            delta  = buffer.rewards[t] + self._gamma * next_v * nont - buffer.values[t]
            gae    = delta + self._gamma * self._gae_lambda * nont * gae
            advantages[t] = gae
            returns[t]    = gae + buffer.values[t]

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
        """PPO update with per-head value losses."""
        T          = len(buffer)
        batch_size = T // self._num_minibatches

        obs_t      = torch.as_tensor(np.stack(buffer.obs), dtype=torch.float32, device=self._device)
        actions_t  = torch.as_tensor(buffer.actions,       dtype=torch.long,    device=self._device)
        old_lp_t   = torch.as_tensor(buffer.log_probs,     dtype=torch.float32, device=self._device)
        adv_t      = torch.as_tensor(advantages,           dtype=torch.float32, device=self._device)
        ret_t      = torch.as_tensor(returns,              dtype=torch.float32, device=self._device)
        weights_t  = torch.as_tensor(np.stack(buffer.persona_weights), dtype=torch.float32, device=self._device)
        ph_tgt_t   = torch.as_tensor(per_persona_returns,              dtype=torch.float32, device=self._device)
        hist_t     = torch.as_tensor(np.stack(buffer.ach_histories),   dtype=torch.long,    device=self._device)  # (T, 5)

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

                logits, value, per_head = self.network(obs_t[idx], weights_t[idx], hist_t[idx])
                dist      = Categorical(logits=logits)
                log_probs = dist.log_prob(actions_t[idx])
                entropy   = dist.entropy().mean()

                ratio  = torch.exp(log_probs - old_lp_t[idx])
                surr1  = ratio * adv_t[idx]
                surr2  = torch.clamp(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = nn.functional.mse_loss(value.squeeze(-1), ret_t[idx])

                mb_tgt = ph_tgt_t[idx]
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