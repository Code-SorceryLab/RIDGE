"""Persona reward functions and RIDGE sigmoid blending engine."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Indices into the state vector produced by CrafterWrapper.extract_state_vector
_IDX_HEALTH = 0
_IDX_FOOD = 1
_IDX_DRINK = 2
_IDX_ENERGY = 3
_IDX_PROGRESS = 4
_IDX_TOOLS = 5

# Fixed weight vectors for baseline personas
_WEIGHTS_EXPLORER: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_WEIGHTS_SURVIVOR: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_WEIGHTS_CRAFTSMAN: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def explorer_reward(info: dict[str, Any]) -> float:
    """Reward for map exploration, new tile discovery, and movement diversity.

    Args:
        info: Enriched info dict from CrafterWrapper step.

    Returns:
        Scalar explorer reward.
    """
    reward = 0.0

    # Reward for cumulative unique tiles visited (dense, small magnitude)
    visited = info.get("visited_count", 0)
    reward += 0.002 * visited

    # Bonus for collecting resources that require exploration
    inv = info.get("inventory", {})
    reward += 0.05 * inv.get("coal", 0)
    reward += 0.10 * inv.get("diamond", 0)

    # Bonus for unlocking exploration-flavoured achievements
    achievements = info.get("achievements", {})
    reward += 0.5 * achievements.get("collect_coal", 0)
    reward += 1.0 * achievements.get("collect_diamond", 0)
    reward += 0.3 * achievements.get("collect_sapling", 0)
    reward += 0.2 * achievements.get("eat_plant", 0)
    reward += 0.2 * achievements.get("collect_drink", 0)

    return float(reward)


def survivor_reward(info: dict[str, Any]) -> float:
    """Reward for maintaining health/food/drink and avoiding damage.

    Args:
        info: Enriched info dict from CrafterWrapper step.

    Returns:
        Scalar survivor reward.
    """
    reward = 0.0

    health = info.get("health", 9) / 9.0
    food = info.get("food", 9) / 9.0
    drink = info.get("drink", 9) / 9.0
    energy = info.get("energy", 9) / 9.0

    # Continuous reward for staying alive with good vitals
    reward += 0.1 * health
    reward += 0.1 * food
    reward += 0.1 * drink
    reward += 0.05 * energy

    # Penalty for critically low vitals
    if health < 0.2:
        reward -= 0.5
    if food < 0.2:
        reward -= 0.3
    if drink < 0.2:
        reward -= 0.3

    # Achievement bonuses aligned with survival
    achievements = info.get("achievements", {})
    reward += 0.5 * achievements.get("collect_drink", 0)
    reward += 0.5 * achievements.get("eat_cow", 0)
    reward += 0.5 * achievements.get("eat_plant", 0)
    reward += 0.3 * achievements.get("wake_up", 0)
    reward += 0.4 * achievements.get("defeat_zombie", 0)
    reward += 0.4 * achievements.get("defeat_skeleton", 0)

    return float(reward)


def craftsman_reward(info: dict[str, Any]) -> float:
    """Reward for crafting achievements, tool creation, and resource collection.

    Args:
        info: Enriched info dict from CrafterWrapper step.

    Returns:
        Scalar craftsman reward.
    """
    reward = 0.0

    # Raw resource inventory (small continuous signal)
    inv = info.get("inventory", {})
    reward += 0.01 * inv.get("wood", 0)
    reward += 0.02 * inv.get("stone", 0)
    reward += 0.03 * inv.get("iron", 0)

    # Achievement bonuses for crafting and placing
    achievements = info.get("achievements", {})
    reward += 0.3 * achievements.get("collect_wood", 0)
    reward += 0.3 * achievements.get("collect_stone", 0)
    reward += 0.5 * achievements.get("collect_iron", 0)
    reward += 0.5 * achievements.get("make_wood_pickaxe", 0)
    reward += 0.5 * achievements.get("make_wood_sword", 0)
    reward += 0.7 * achievements.get("make_stone_pickaxe", 0)
    reward += 0.7 * achievements.get("make_stone_sword", 0)
    reward += 1.0 * achievements.get("make_iron_pickaxe", 0)
    reward += 1.0 * achievements.get("make_iron_sword", 0)
    reward += 0.4 * achievements.get("place_table", 0)
    reward += 0.4 * achievements.get("place_furnace", 0)
    reward += 0.2 * achievements.get("place_stone", 0)
    reward += 0.2 * achievements.get("place_plant", 0)

    return float(reward)


def sigma(state_vector: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Compute smooth sigmoid-based persona weights from the game state.

    Weights are computed from three soft conditions using sigmoid activations,
    then L1-normalised to sum to 1. No hard switches.

    State vector layout (from CrafterWrapper.extract_state_vector):
      [0] health_norm, [1] food_norm, [2] drink_norm, [3] energy_norm,
      [4] progress_norm, [5] tool_progress

    Args:
        state_vector: Float32 ndarray of shape (6,).
        config: Config dict containing sigmoid_temperature, health_threshold,
            hunger_threshold, and progress_threshold.

    Returns:
        Float32 ndarray of shape (3,) — [w_explorer, w_survivor, w_craftsman],
        summing to 1.
    """
    temp: float = float(config.get("sigmoid_temperature", 1.0))
    h_thresh: float = float(config.get("health_threshold", 0.3))
    f_thresh: float = float(config.get("hunger_threshold", 0.3))
    p_thresh: float = float(config.get("progress_threshold", 0.5))

    health = float(state_vector[_IDX_HEALTH])
    food = float(state_vector[_IDX_FOOD])
    drink = float(state_vector[_IDX_DRINK])
    progress = float(state_vector[_IDX_PROGRESS])

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x / temp))

    # Survivor drive: high when health or food/drink is critically low
    survivor_drive = _sigmoid(h_thresh - health) + _sigmoid(f_thresh - food) + _sigmoid(f_thresh - drink)

    # Craftsman drive: high when basic survival is secure and progress is lagging
    survival_secure = _sigmoid(health - h_thresh) * _sigmoid(food - f_thresh)
    craftsman_drive = survival_secure * _sigmoid(p_thresh - progress)

    # Explorer drive: high when secure and progress is advancing
    explorer_drive = survival_secure * _sigmoid(progress - p_thresh * 0.5)

    weights = np.array([explorer_drive, survivor_drive, craftsman_drive], dtype=np.float32)

    total = weights.sum()
    if total < 1e-8:
        # Fallback: uniform weights if all drives collapse
        weights = np.ones(3, dtype=np.float32) / 3.0
    else:
        weights /= total

    return weights  # (3,)


def compute_blended_reward(
    info: dict[str, Any],
    state_vector: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Apply RIDGE blending to produce a scalar reward for the PPO update.

    In fixed persona modes, weights are locked and the blended reward reduces
    to the single active persona reward.

    Args:
        info: Enriched info dict from CrafterWrapper step.
        state_vector: Float32 ndarray of shape (6,) from extract_state_vector.
        config: Config dict specifying blending_mode and sigmoid parameters.

    Returns:
        Tuple of:
          - blended_reward (float): Scalar reward fed to PPO.
          - weights (np.ndarray): Shape (3,) persona weights used this step.
          - per_persona (dict): Raw rewards keyed 'explorer', 'survivor', 'craftsman'.
    """
    mode: str = config.get("blending_mode", "ridge")

    r_e = explorer_reward(info)
    r_s = survivor_reward(info)
    r_c = craftsman_reward(info)
    per_persona = {"explorer": r_e, "survivor": r_s, "craftsman": r_c}

    if mode == "fixed_explorer":
        weights = _WEIGHTS_EXPLORER.copy()
    elif mode == "fixed_survivor":
        weights = _WEIGHTS_SURVIVOR.copy()
    elif mode == "fixed_craftsman":
        weights = _WEIGHTS_CRAFTSMAN.copy()
    else:
        weights = sigma(state_vector, config)

    blended = float(weights[0] * r_e + weights[1] * r_s + weights[2] * r_c)
    return blended, weights, per_persona
