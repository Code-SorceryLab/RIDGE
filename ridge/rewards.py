"""Persona reward functions and RIDGE sigmoid blending engine — 4 personas."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Indices into the state vector produced by CrafterWrapper.extract_state_vector
_IDX_HEALTH  = 0
_IDX_FOOD    = 1
_IDX_DRINK   = 2
_IDX_ENERGY  = 3
_IDX_PROGRESS = 4
_IDX_TOOLS   = 5

# Fixed weight vectors for baseline personas — shape (4,): [explorer, survivor, craftsman, warrior]
_WEIGHTS_EXPLORER  = np.array([1., 0., 0., 0.], dtype=np.float32)
_WEIGHTS_SURVIVOR  = np.array([0., 1., 0., 0.], dtype=np.float32)
_WEIGHTS_CRAFTSMAN = np.array([0., 0., 1., 0.], dtype=np.float32)
_WEIGHTS_WARRIOR   = np.array([0., 0., 0., 1.], dtype=np.float32)

# Canonical persona order — must match RIDGENetwork head order in agent.py
PERSONA_NAMES = ["explorer", "survivor", "craftsman", "warrior"]

# ─────────────────────────────────────────────────────────────────────────────
#  Achievement bonus tables  (one-time per episode per persona)
#
#  Split rationale:
#    Explorer  — broad collection, discovery, movement
#    Survivor  — vitals, sleep, staying alive
#    Craftsman — tech tree: tools, resources, stations
#    Warrior   — weapons and kills (previously split across survivor+craftsman,
#                causing the blender conflict where combat required being in
#                two personas simultaneously)
#
#  Scale: easy 0.5–1.0 | mid 2.0–3.0 | iron tier 4.0–5.0 | diamond 8.0
# ─────────────────────────────────────────────────────────────────────────────

_EXPLORER_BONUSES = {
    "collect_wood":    0.5,
    "collect_stone":   0.5,
    "collect_coal":    1.5,
    "collect_sapling": 1.0,
    "collect_drink":   0.5,
    "collect_iron":    2.0,
    "collect_diamond": 6.0,
    "eat_plant":       0.5,
    "place_plant":     0.5,
    "wake_up":         0.3,
}

_SURVIVOR_BONUSES = {
    "collect_drink":   1.0,
    "eat_cow":         1.5,
    "eat_plant":       1.0,
    "wake_up":         1.5,
}

_CRAFTSMAN_BONUSES = {
    "collect_wood":       0.5,
    "collect_stone":      0.5,
    "place_table":        1.0,
    "make_wood_pickaxe":  1.5,
    "collect_coal":       2.0,
    "place_furnace":      2.5,
    "make_stone_pickaxe": 2.5,
    "collect_iron":       4.0,
    "make_iron_pickaxe":  5.0,
    "collect_diamond":    8.0,
    "wake_up":            0.5,
}

_WARRIOR_BONUSES = {
    "collect_wood":       0.3,   # need wood to start weapon chain
    "place_table":        0.5,
    "make_wood_sword":    2.0,
    "collect_stone":      0.3,
    "make_stone_sword":   3.0,
    "collect_coal":       1.0,
    "collect_iron":       2.0,
    "place_furnace":      1.5,
    "make_iron_sword":    6.0,   # top priority
    "defeat_zombie":      4.0,
    "defeat_skeleton":    5.0,
    "eat_cow":            0.5,
    "eat_plant":          0.3,
    "wake_up":            0.4,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Persona reward functions
# ─────────────────────────────────────────────────────────────────────────────

def explorer_reward(info: dict[str, Any], unlocked: set[str]) -> float:
    """Reward exploration: new tile discovery + broad resource collection."""
    reward = 0.0
    reward += 0.10 * info.get("delta_visited", 0)
    energy = info.get("energy", 9) / 9.0
    if energy < 0.3: reward -= 0.2
    for name, bonus in _EXPLORER_BONUSES.items():
        if info.get("achievements", {}).get(name, 0) and name not in unlocked:
            unlocked.add(name)
            reward += bonus
    return float(reward)


def survivor_reward(info: dict[str, Any], unlocked: set[str]) -> float:
    """Reward survival: continuous vital shaping + food/sleep milestones."""
    reward = 0.0
    health = info.get("health", 9) / 9.0
    food   = info.get("food",   9) / 9.0
    drink  = info.get("drink",  9) / 9.0
    energy = info.get("energy", 9) / 9.0

    reward += 0.10 * health
    reward += 0.10 * food
    reward += 0.10 * drink
    reward += 0.05 * energy

    if health < 0.2: reward -= 0.5
    if food   < 0.2: reward -= 0.3
    if drink  < 0.2: reward -= 0.3

    for name, bonus in _SURVIVOR_BONUSES.items():
        if info.get("achievements", {}).get(name, 0) and name not in unlocked:
            unlocked.add(name)
            reward += bonus
    return float(reward)


def craftsman_reward(info: dict[str, Any], unlocked: set[str]) -> float:
    """Reward crafting: inventory stockpile + tech-tree milestones."""
    reward = 0.0
    energy = info.get("energy", 9) / 9.0
    reward += 0.05 * energy
    if energy < 0.3: reward -= 0.3
    inv = info.get("inventory", {})
    reward += 0.005 * min(inv.get("wood",    0), 10)
    reward += 0.008 * min(inv.get("stone",   0), 10)
    reward += 0.015 * min(inv.get("coal",    0),  8)
    reward += 0.030 * min(inv.get("iron",    0),  6)
    reward += 0.060 * min(inv.get("diamond", 0),  3)

    for name, bonus in _CRAFTSMAN_BONUSES.items():
        if info.get("achievements", {}).get(name, 0) and name not in unlocked:
            unlocked.add(name)
            reward += bonus
    return float(reward)


def warrior_reward(info: dict[str, Any], unlocked: set[str]) -> float:
    """Reward combat: weapon crafting + enemy kills.

    Dense per-step signal from health delta — warriors take hits but
    kills are the primary objective so health loss is penalised lightly.
    """
    reward = 0.0

    # Light health shaping — warriors expect damage, don't over-penalise
    health = info.get("health", 9) / 9.0
    energy = info.get("energy", 9) / 9.0
    reward += 0.05 * health
    if energy < 0.3: reward -= 0.2

    for name, bonus in _WARRIOR_BONUSES.items():
        if info.get("achievements", {}).get(name, 0) and name not in unlocked:
            unlocked.add(name)
            reward += bonus
    return float(reward)


# ─────────────────────────────────────────────────────────────────────────────
#  Sigma — 4-persona state-conditioned blender
#
#  Output order: [w_explorer, w_survivor, w_craftsman, w_warrior]
#
#  Signals:
#    Explorer  — (0.10 - progress): strong early, yields after ~2 achievements
#    Survivor  — need = 1 - min(health, food, drink): live every step
#    Craftsman — tool_progress [0,1]: rises through all crafting tiers
#    Warrior   — weapon_tier [0,1] + kill bonus: activates once armed
#
#  weapon_tier is derived from tool_progress as a proxy. It's not perfect
#  but avoids adding a new state vector dimension. Values:
#    tool_progress ≈ 0.17  →  wood sword crafted  (1/6 tools)
#    tool_progress ≈ 0.50  →  stone sword crafted (3/6 tools)
#    tool_progress ≈ 0.83  →  iron sword crafted  (5/6 tools)
# ─────────────────────────────────────────────────────────────────────────────

def sigma(state_vector: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Compute 4 normalised persona weights from live game state.

    Args:
        state_vector: Float32 ndarray of shape (6,).
        config: Config dict with:
            sigmoid_temperature : global temperature divisor (legacy, kept for compat)
            health_threshold    : need level at which survivor activates (default 0.3)
            weapon_threshold    : tool_progress at which warrior activates (default 0.15)
            blend_sharpness     : multiplier on all sigmoid slopes (default 1.0).
                                  >1 = harder, more switch-like transitions.
                                  <1 = softer, more uniform blending.
                                  This is the primary RQ3 ablation parameter.

    Returns:
        Float32 ndarray of shape (4,) — [w_explorer, w_survivor, w_craftsman, w_warrior],
        summing to 1.
    """
    temp      = float(config.get("sigmoid_temperature", 1.0))
    h_thresh  = float(config.get("health_threshold", 0.3))
    w_thresh  = float(config.get("weapon_threshold", 0.15))
    sharpness = float(config.get("blend_sharpness", 1.0))

    health        = float(state_vector[_IDX_HEALTH])
    food          = float(state_vector[_IDX_FOOD])
    drink         = float(state_vector[_IDX_DRINK])
    progress      = float(state_vector[_IDX_PROGRESS])
    tool_progress = float(state_vector[_IDX_TOOLS])

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-max(-20., min(20., x / temp))))

    # Explorer: strong early, fades as agent specialises
    w_e = _sigmoid(4.0 * sharpness * (0.10 - progress))

    # Survivor: driven by live need signal — varies every step
    need  = 1.0 - min(health, food, drink)
    w_s   = _sigmoid(8.0 * sharpness * (need - h_thresh))

    # Craftsman: rises through all tool tiers, never collapses
    w_c = _sigmoid(6.0 * sharpness * (tool_progress - 0.15))

    # Warrior: activates once a sword is crafted (tool_progress > w_thresh)
    w_w = _sigmoid(7.0 * sharpness * (tool_progress - w_thresh))

    weights = np.array([w_e, w_s, w_c, w_w], dtype=np.float32)
    total = weights.sum()
    if total < 1e-8:
        weights = np.ones(4, dtype=np.float32) / 4.0
    else:
        weights /= total
    return weights  # (4,)


# ─────────────────────────────────────────────────────────────────────────────
#  Blended reward entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_blended_reward(
    info: dict[str, Any],
    state_vector: np.ndarray,
    config: dict[str, Any],
    episode_unlocked: dict[str, set[str]],
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Blend 4 persona rewards into a scalar for PPO.

    Args:
        info: Enriched info dict from CrafterWrapper step.
        state_vector: Float32 ndarray shape (6,).
        config: Config dict.
        episode_unlocked: Per-persona sets of already-claimed achievements.
            Keys: 'explorer', 'survivor', 'craftsman', 'warrior'.
            Caller resets on each episode reset.

    Returns:
        (blended_reward, weights (4,), per_persona dict)
    """
    mode: str = config.get("blending_mode", "ridge")

    r_e = explorer_reward (info, episode_unlocked["explorer"])
    r_s = survivor_reward (info, episode_unlocked["survivor"])
    r_c = craftsman_reward(info, episode_unlocked["craftsman"])
    r_w = warrior_reward  (info, episode_unlocked["warrior"])
    per_persona = {"explorer": r_e, "survivor": r_s, "craftsman": r_c, "warrior": r_w}

    if   mode == "fixed_explorer":  weights = _WEIGHTS_EXPLORER.copy()
    elif mode == "fixed_survivor":  weights = _WEIGHTS_SURVIVOR.copy()
    elif mode == "fixed_craftsman": weights = _WEIGHTS_CRAFTSMAN.copy()
    elif mode == "fixed_warrior":   weights = _WEIGHTS_WARRIOR.copy()
    else:                           weights = sigma(state_vector, config)

    blended = float(
        weights[0] * r_e + weights[1] * r_s + weights[2] * r_c + weights[3] * r_w
    )

    # Unconditional sleep bonus — outside persona blending so it's never
    # diluted by weight shifts. Fires once per episode regardless of mode.
    global_unlocked = episode_unlocked.setdefault("_global", set())
    if info.get("achievements", {}).get("wake_up", 0) and "wake_up" not in global_unlocked:
        global_unlocked.add("wake_up")
        blended += 0.5

    return blended, weights, per_persona