"""
sharpness_sweep.py

Trains one RIDGE agent per blend_sharpness value and records:
  - mean/std episode reward
  - achievement unlock rates
  - per-persona weight trajectories (mean ± std per episode)
  - PPO value losses per head (proxy for training stability — RQ3)

Results are written to:
  results/sharpness_sweep/sharpness_{value}.json
  results/sharpness_sweep/summary.json   ← all conditions in one file for plotting

Usage:
  python scripts/sharpness_sweep.py
  python scripts/sharpness_sweep.py --sharpness 0.25 0.5 1.0 2.0 4.0
  python scripts/sharpness_sweep.py --sharpness 0.25 0.5 1.0 2.0 4.0 --steps 300000 --seed 42
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ridge.agent import PPOAgent, RolloutBuffer
from ridge.game import make_env
from ridge.rewards import compute_blended_reward
from ridge.utils import ensure_dir, get_device, load_default_config, set_seeds, setup_logging

logger = logging.getLogger(__name__)

# Default sharpness values to sweep — covers near-uniform → hard-switch
DEFAULT_SHARPNESS = [0.10, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]

ALL_ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]

PERSONA_NAMES = ["explorer", "survivor", "craftsman", "warrior"]


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run metrics collector
# ─────────────────────────────────────────────────────────────────────────────

class SweepCollector:
    """Accumulates per-episode metrics during a single sharpness training run."""

    def __init__(self) -> None:
        self.episode_rewards:      list[float]       = []
        self.episode_lengths:      list[int]         = []
        self.episode_achievements: list[list[str]]   = []
        self.achievement_counts:   dict[str, int]    = defaultdict(int)

        # Per-step weight history — flushed to episode_weight_means at episode end
        self._step_weights:        list[np.ndarray]  = []   # (4,) each step
        self.episode_weight_means: list[np.ndarray]  = []   # mean (4,) per episode
        self.episode_weight_stds:  list[np.ndarray]  = []   # std  (4,) per episode

        # Value loss per head per update — for RQ3 stability analysis
        self.value_losses: dict[str, list[float]] = {p: [] for p in PERSONA_NAMES}
        self.policy_losses: list[float] = []

    def record_step_weights(self, weights: np.ndarray) -> None:
        self._step_weights.append(weights.copy())

    def record_episode_end(self, reward: float, length: int, achievements: list[str]) -> None:
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_achievements.append(achievements)
        for name in achievements:
            self.achievement_counts[name] += 1

        if self._step_weights:
            arr = np.stack(self._step_weights)   # (T, 4)
            self.episode_weight_means.append(arr.mean(axis=0))
            self.episode_weight_stds.append(arr.std(axis=0))
        self._step_weights = []

    def record_update(self, metrics: dict[str, float]) -> None:
        for p in PERSONA_NAMES:
            key = f"value_loss_{p}"
            if key in metrics:
                self.value_losses[p].append(metrics[key])
        if "policy_loss" in metrics:
            self.policy_losses.append(metrics["policy_loss"])

    def to_dict(self, sharpness: float) -> dict[str, Any]:
        n = len(self.episode_rewards)
        unlock_rates = {
            ach: self.achievement_counts.get(ach, 0) / n * 100 if n else 0.0
            for ach in ALL_ACHIEVEMENTS
        }
        n_ach_per_ep = [len(ep) for ep in self.episode_achievements]

        # Weight trajectory summary
        weight_summary = {}
        if self.episode_weight_means:
            means_arr = np.stack(self.episode_weight_means)  # (episodes, 4)
            stds_arr  = np.stack(self.episode_weight_stds)
            for i, p in enumerate(PERSONA_NAMES):
                weight_summary[p] = {
                    "mean_weight":      float(means_arr[:, i].mean()),
                    "std_weight":       float(means_arr[:, i].std()),
                    "mean_intra_std":   float(stds_arr[:, i].mean()),  # within-episode variation
                }

        # Value loss stability — mean and std over all updates
        value_loss_summary = {}
        for p in PERSONA_NAMES:
            vals = self.value_losses[p]
            value_loss_summary[p] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std":  float(np.std(vals))  if vals else 0.0,
                "final_100_mean": float(np.mean(vals[-100:])) if len(vals) >= 100 else
                                  float(np.mean(vals)) if vals else 0.0,
            }

        return {
            "blend_sharpness":        sharpness,
            "n_episodes":             n,
            "mean_reward":            float(np.mean(self.episode_rewards))  if n else 0.0,
            "std_reward":             float(np.std(self.episode_rewards))   if n else 0.0,
            "mean_achievements":      float(np.mean(n_ach_per_ep))          if n else 0.0,
            "std_achievements":       float(np.std(n_ach_per_ep))           if n else 0.0,
            "achievement_unlock_rates": unlock_rates,
            "weight_summary":         weight_summary,
            "value_loss_summary":     value_loss_summary,
            "score_history":          self.episode_rewards,
            "achievement_history":    n_ach_per_ep,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Single training run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    config: dict[str, Any],
    sharpness: float,
    total_steps: int,
    seed: int,
    out_dir: str,
) -> dict[str, Any]:
    """Train one RIDGE agent with a fixed blend_sharpness and return metrics."""

    run_config = dict(config)
    run_config["blend_sharpness"] = sharpness
    run_config["blending_mode"]   = "ridge"
    run_config["seed"]            = seed

    set_seeds(seed)
    device  = get_device()
    env     = make_env(run_config, seed=seed)
    agent   = PPOAgent(run_config, num_actions=env.action_space.n, device=device)
    buffer  = RolloutBuffer()
    collect = SweepCollector()

    rollout_steps = int(run_config.get("rollout_steps", 512))

    obs, info        = env.reset()
    weights          = np.full(4, 0.25, dtype=np.float32)
    episode_unlocked = {"explorer": set(), "survivor": set(),
                        "craftsman": set(), "warrior": set()}
    ep_reward        = 0.0
    ep_steps         = 0
    ep_achievements: list[str] = []
    global_step      = 0

    import torch
    from tqdm import tqdm
    pbar = tqdm(
        total=total_steps, unit="step", dynamic_ncols=True,
        desc=f"sharpness={sharpness:.2f}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )

    while global_step < total_steps:
        buffer.clear()

        for _ in range(rollout_steps):
            state_vec = env.extract_state_vector(info)
            blended, weights, per_persona = compute_blended_reward(
                info, state_vec, run_config, episode_unlocked
            )

            action, log_prob, value, per_head_val = agent.select_action(obs, weights)
            next_obs, _, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            per_persona_arr = np.array([
                per_persona["explorer"], per_persona["survivor"],
                per_persona["craftsman"], per_persona["warrior"],
            ], dtype=np.float32)

            buffer.obs.append(obs.copy())
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob)
            buffer.values.append(value)
            buffer.per_head_values.append(per_head_val)
            buffer.persona_step_rewards.append(per_persona_arr)
            buffer.rewards.append(blended)
            buffer.dones.append(done)
            buffer.persona_weights.append(weights.copy())
            buffer.infos.append(next_info)

            collect.record_step_weights(weights)
            ep_reward += blended
            ep_steps  += 1

            # Track achievements for this episode
            for name in ALL_ACHIEVEMENTS:
                if next_info.get("achievements", {}).get(name, 0) and name not in ep_achievements:
                    ep_achievements.append(name)

            global_step += 1

            if done:
                collect.record_episode_end(ep_reward, ep_steps, list(ep_achievements))
                obs, info        = env.reset()
                weights          = np.full(4, 0.25, dtype=np.float32)
                episode_unlocked = {"explorer": set(), "survivor": set(),
                                    "craftsman": set(), "warrior": set()}
                ep_reward        = 0.0
                ep_steps         = 0
                ep_achievements  = []
            else:
                obs  = next_obs
                info = next_info

            if global_step >= total_steps:
                break

        # PPO update
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            w_t   = torch.as_tensor(weights, dtype=torch.float32, device=device)
            _, last_val_t, last_ph_t = agent.network(obs_t, w_t)
            last_val = float(last_val_t.squeeze().item())
            last_ph  = last_ph_t.squeeze(0).cpu().numpy()

        advantages, returns, per_persona_returns = agent.compute_advantages(
            buffer, last_val, last_ph
        )
        metrics = agent.ppo_update(buffer, advantages, returns, per_persona_returns)
        collect.record_update(metrics)

        n_ep = len(collect.episode_rewards)
        recent = collect.episode_rewards[-20:] if n_ep >= 20 else collect.episode_rewards
        pbar.update(min(rollout_steps, total_steps - (global_step - rollout_steps)))
        pbar.set_postfix({
            "ep": n_ep,
            "r": f"{np.mean(recent):.1f}" if recent else "—",
            "vl_w": f"{metrics.get('value_loss_warrior', 0):.3f}",
        })

    pbar.close()
    env.close()

    result = collect.to_dict(sharpness)

    path = Path(out_dir) / f"sharpness_{sharpness}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("sharpness=%.2f  mean_reward=%.2f  mean_achievements=%.1f  saved→%s",
                sharpness, result["mean_reward"], result["mean_achievements"], path)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    sharpness_values: list[float],
    config: dict[str, Any],
    total_steps: int,
    seed: int,
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    all_results = []

    print(f"\nSharpness sweep: {sharpness_values}")
    print(f"Steps per condition: {total_steps:,}  |  seed: {seed}")
    print(f"Output: {out_dir}\n")

    for sharpness in sharpness_values:
        print(f"\n{'='*60}")
        print(f"  blend_sharpness = {sharpness}")
        print(f"{'='*60}")
        result = run_one(config, sharpness, total_steps, seed, out_dir)
        all_results.append(result)

    # Save combined summary
    summary_path = Path(out_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSweep complete. Summary → {summary_path}")

    # Quick console table
    print(f"\n{'sharpness':>10} {'mean_r':>8} {'mean_ach':>10} {'vl_warrior_final':>18}")
    print("-" * 52)
    for r in all_results:
        vl_w = r["value_loss_summary"]["warrior"]["final_100_mean"]
        print(f"{r['blend_sharpness']:>10.2f} {r['mean_reward']:>8.2f} "
              f"{r['mean_achievements']:>10.2f} {vl_w:>18.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep blend_sharpness for RQ3")
    parser.add_argument("--config",    type=str,   default="configs/ridge_blend.yaml")
    parser.add_argument("--sharpness", type=float, nargs="+", default=DEFAULT_SHARPNESS)
    parser.add_argument("--steps",     type=int,   default=500_000,
                        help="Training steps per condition (default 500k for sweep speed)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--out",       type=str,   default="results/sharpness_sweep")
    args = parser.parse_args()

    setup_logging()
    config = load_default_config(args.config)
    run_sweep(
        sharpness_values=sorted(args.sharpness),
        config=config,
        total_steps=args.steps,
        seed=args.seed,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
