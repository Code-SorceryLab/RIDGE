"""Crafter environment wrapper, observation processing, and state extraction."""

import logging
from typing import Any

import crafter
import numpy as np

logger = logging.getLogger(__name__)

# All 22 Crafter achievement names in canonical order
ACHIEVEMENTS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]

# Inventory item names tracked for the state vector
INVENTORY_KEYS = ["wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe",
                  "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]


class EpisodeStats:
    """Tracks per-episode statistics for logging and viewer display."""

    def __init__(self) -> None:
        self.total_reward: float = 0.0
        self.steps: int = 0
        self.achievements_unlocked: list[str] = []
        self.persona_weights: list[np.ndarray] = []  # list of (3,) arrays
        self.score: float = 0.0

    def update(self, reward: float, info: dict[str, Any], weights: np.ndarray) -> None:
        """Record one environment step.

        Args:
            reward: Scalar reward received this step.
            info: Info dict from Crafter step.
            weights: Current persona weights array of shape (3,).
        """
        self.total_reward += reward
        self.steps += 1
        self.persona_weights.append(weights.copy())

        for name in ACHIEVEMENTS:
            if info.get("achievements", {}).get(name, 0) and name not in self.achievements_unlocked:
                self.achievements_unlocked.append(name)
                logger.info("Achievement unlocked: %s", name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize episode stats to a plain dict.

        Returns:
            Dict with total_reward, steps, achievements_unlocked, score,
            and mean_weights.
        """
        weights_arr = np.stack(self.persona_weights) if self.persona_weights else np.zeros((1, 3))
        return {
            "total_reward": self.total_reward,
            "steps": self.steps,
            "achievements_unlocked": list(self.achievements_unlocked),
            "achievement_count": len(self.achievements_unlocked),
            "score": self.score,
            "mean_weights": weights_arr.mean(axis=0).tolist(),
        }


class CrafterWrapper:
    """Wraps the Crafter environment with observation processing and state extraction.

    Processes 64×64 RGB observations, extracts the internal state vector used
    by RIDGE's sigmoid blending engine, and tracks episode-level statistics.
    """

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        """Initialise the wrapper and underlying Crafter env.

        Args:
            config: Project config dict (loaded from YAML).
            seed: Optional random seed for the environment.
        """
        self._config = config
        self._seed = seed
        self._env = crafter.Env()
        # if seed is not None:
        #     self._env.seed(seed)

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._episode_stats = EpisodeStats()
        self._prev_achievements: dict[str, int] = {}
        self._visited_positions: set[tuple[int, int]] = set()
        self._last_pos: tuple[int, int] | None = None
        self._steps_this_episode: int = 0

        logger.debug("CrafterWrapper initialised (seed=%s)", seed)

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return processed observation and info.

        Returns:
            Tuple of (processed_obs, info_dict).
        """
        obs = self._env.reset()
        self._episode_stats = EpisodeStats()
        self._prev_achievements = {}
        self._visited_positions = set()
        self._last_pos = None
        self._steps_this_episode = 0

        info: dict[str, Any] = {
            "achievements": {name: 0 for name in ACHIEVEMENTS},
            "health": 9,
            "food": 9,
            "drink": 9,
            "energy": 9,
            "inventory": {k: 0 for k in INVENTORY_KEYS},
            "player_pos": (0, 0),
        }
        return self._process_obs(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment and return processed outputs.

        Args:
            action: Integer action index.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, done, info = self._env.step(action)
        self._steps_this_episode += 1

        info = self._enrich_info(info)
        processed_obs = self._process_obs(obs)

        terminated = done
        truncated = False
        return processed_obs, float(reward), terminated, truncated, info

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize and format a raw Crafter observation.

        Args:
            obs: Raw RGB observation of shape (H, W, 3), uint8.

        Returns:
            Float32 array of shape (3, H, W), values in [0, 1].
        """
        # (H, W, C) → (C, H, W), float32 in [0, 1]
        obs = obs.astype(np.float32) / 255.0
        return np.transpose(obs, (2, 0, 1))  # (3, 64, 64)

    def _enrich_info(self, raw_info: dict[str, Any]) -> dict[str, Any]:
        """Augment the raw Crafter info dict with derived fields.

        Args:
            raw_info: Info dict returned directly by Crafter step.

        Returns:
            Enriched info dict with normalised vitals, inventory, and position.
        """
        info: dict[str, Any] = dict(raw_info)

        # Crafter info keys vary by version — pull with safe defaults
        achievements = info.get("achievements", {})
        if not isinstance(achievements, dict):
            achievements = {}
        info["achievements"] = {name: int(achievements.get(name, 0)) for name in ACHIEVEMENTS}

        player = info.get("player", {})
        if not isinstance(player, dict):
            player = {}

        info["health"] = int(player.get("health", 9))
        info["food"] = int(player.get("food", 9))
        info["drink"] = int(player.get("drink", 9))
        info["energy"] = int(player.get("energy", 9))

        inventory = player.get("inventory", {})
        if not isinstance(inventory, dict):
            inventory = {}
        info["inventory"] = {k: int(inventory.get(k, 0)) for k in INVENTORY_KEYS}

        pos = player.get("pos", (0, 0))
        info["player_pos"] = tuple(pos) if hasattr(pos, "__iter__") else (0, 0)

        # Track visited tiles for explorer reward
        self._visited_positions.add(info["player_pos"])
        info["visited_count"] = len(self._visited_positions)

        info["steps_this_episode"] = self._steps_this_episode
        return info

    def extract_state_vector(self, info: dict[str, Any]) -> np.ndarray:
        """Build the normalised state vector fed to RIDGE's sigmoid blending.

        State vector components (all in [0, 1]):
          [0] health_norm     — health / 9
          [1] food_norm       — food / 9
          [2] drink_norm      — drink / 9
          [3] energy_norm     — energy / 9
          [4] progress_norm   — achievements unlocked / 22
          [5] tool_progress   — total tools crafted (capped at 6) / 6

        Args:
            info: Enriched info dict from a step call.

        Returns:
            Float32 ndarray of shape (6,).
        """
        health_norm = info["health"] / 9.0
        food_norm = info["food"] / 9.0
        drink_norm = info["drink"] / 9.0
        energy_norm = info["energy"] / 9.0

        achievement_count = sum(info["achievements"].values())
        progress_norm = achievement_count / len(ACHIEVEMENTS)

        inv = info["inventory"]
        tools = (inv.get("wood_pickaxe", 0) + inv.get("stone_pickaxe", 0) +
                 inv.get("iron_pickaxe", 0) + inv.get("wood_sword", 0) +
                 inv.get("stone_sword", 0) + inv.get("iron_sword", 0))
        tool_progress = min(tools, 6) / 6.0

        return np.array(
            [health_norm, food_norm, drink_norm, energy_norm, progress_norm, tool_progress],
            dtype=np.float32,
        )  # (6,)

    def update_episode_stats(self, reward: float, info: dict[str, Any], weights: np.ndarray) -> None:
        """Accumulate stats for the current episode.

        Args:
            reward: Scalar reward this step.
            info: Enriched info dict.
            weights: Persona weights array of shape (3,).
        """
        self._episode_stats.update(reward, info, weights)

    def get_episode_stats(self) -> EpisodeStats:
        """Return the stats object for the current episode.

        Returns:
            EpisodeStats instance for the ongoing episode.
        """
        return self._episode_stats

    def seed(self, seed: int) -> None:
        """Re-seed the underlying environment.

        Args:
            seed: New seed value.
        """
        self._seed = seed
        self._env.seed(seed)

    def render(self) -> np.ndarray | None:
        """Render the current frame as an RGB array.

        Returns:
            uint8 RGB array of shape (H, W, 3), or None if unavailable.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the underlying Crafter environment."""
        self._env.close()


def make_env(config: dict[str, Any], seed: int | None = None) -> CrafterWrapper:
    """Construct and return a CrafterWrapper.

    Args:
        config: Project config dict.
        seed: Optional random seed.

    Returns:
        Configured CrafterWrapper instance.
    """
    env = CrafterWrapper(config=config, seed=seed)
    logger.info("Created CrafterWrapper (seed=%s)", seed)
    return env
