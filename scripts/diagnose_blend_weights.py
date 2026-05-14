"""Diagnose RIDGE blender normalisation on a survivor_baseline trajectory.

Runs the trained survivor agent for one or more episodes, and at every step
logs both the raw pre-normalisation sigmoid activations and the normalised
weights the RIDGE blender would emit. Detects whether softmax-style
normalisation washes out Survivor's signal when other personas saturate.

Output:
    diagnostics/<run>/weights_per_step.csv        — per-step values
    diagnostics/<run>/blend_weights_diagnostic.png/.pdf — 3-panel plot
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def run_diagnostic(
    config: dict[str, Any],
    checkpoint_path: str,
    n_episodes: int,
    seed: int,
    out_dir: str,
) -> None:
    """Run the diagnostic rollout and emit CSV + plot.

    The trained survivor agent is driven with its training-time fixed weights
    [0, 1, 0, 0]; the RIDGE blender is computed in shadow at every step so we
    can compare raw vs normalised Survivor weight on a realistic trajectory.
    """
    from ridge.agent import PPOAgent
    from ridge.game import make_env
    from ridge.rewards import _compute_raw_sigmoids, sigma
    from ridge.utils import get_device, set_seeds, setup_logging

    setup_logging()
    set_seeds(seed)
    device = get_device()

    env = make_env(config, seed=seed)
    agent = PPOAgent(config, num_actions=env.action_space.n, device=device)
    agent.load_checkpoint(checkpoint_path)
    agent.network.eval()

    fixed_survivor = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            state_vec = env.extract_state_vector(info)
            raw = _compute_raw_sigmoids(state_vec, config)
            norm = sigma(state_vec, config)
            rows.append({
                "episode":       ep,
                "step":          step,
                "health":        float(state_vec[0]),
                "food":          float(state_vec[1]),
                "drink":         float(state_vec[2]),
                "energy":        float(state_vec[3]),
                "progress":      float(state_vec[4]),
                "tool_progress": float(state_vec[5]),
                "raw_w_e":       float(raw[0]),
                "raw_w_s":       float(raw[1]),
                "raw_w_c":       float(raw[2]),
                "raw_w_w":       float(raw[3]),
                "norm_w_e":      float(norm[0]),
                "norm_w_s":      float(norm[1]),
                "norm_w_c":      float(norm[2]),
                "norm_w_w":      float(norm[3]),
            })
            action, _, _, _ = agent.select_action(obs, fixed_survivor)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        logger.info("Episode %d/%d done — %d steps", ep + 1, n_episodes, step)

    csv_path = out / "weights_per_step.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV saved to %s", csv_path)

    raw_w_s  = np.array([r["raw_w_s"]  for r in rows])
    norm_w_s = np.array([r["norm_w_s"] for r in rows])
    raw_w_c  = np.array([r["raw_w_c"]  for r in rows])
    raw_w_w  = np.array([r["raw_w_w"]  for r in rows])
    energy   = np.array([r["energy"]   for r in rows])
    health   = np.array([r["health"]   for r in rows])
    food     = np.array([r["food"]     for r in rows])
    drink    = np.array([r["drink"]    for r in rows])
    need     = 1.0 - np.minimum.reduce([health, food, drink, energy])

    print(f"\n--- Survivor weight summary — {len(rows)} steps across {n_episodes} episode(s) ---")
    print(f"  raw_w_s   mean={raw_w_s.mean():.4f}  max={raw_w_s.max():.4f}  p95={np.percentile(raw_w_s, 95):.4f}")
    print(f"  norm_w_s  mean={norm_w_s.mean():.4f}  max={norm_w_s.max():.4f}  p95={np.percentile(norm_w_s, 95):.4f}")
    print(f"  energy    mean={energy.mean():.4f}   min={energy.min():.4f}")
    print(f"  need      mean={need.mean():.4f}    max={need.max():.4f}")
    print(f"  raw_w_c   mean={raw_w_c.mean():.4f}  max={raw_w_c.max():.4f}  (saturation check)")
    print(f"  raw_w_w   mean={raw_w_w.mean():.4f}  max={raw_w_w.max():.4f}  (saturation check)")

    suppressed = (raw_w_s > 0.4) & (norm_w_s < 0.1)
    if suppressed.any():
        n = int(suppressed.sum())
        ratio = (raw_w_s[suppressed] / np.maximum(norm_w_s[suppressed], 1e-6)).max()
        print(f"\n  *** NORMALISATION ARTIFACT *** ")
        print(f"  {n}/{len(rows)} steps have raw_w_s > 0.4 but norm_w_s < 0.1")
        print(f"  Max suppression ratio: {ratio:.1f}x")
    else:
        print("\n  No suppression detected — when Survivor's raw signal rises, normalised weight rises with it.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping plot")
        return

    steps = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(steps, [r["raw_w_e"] for r in rows], label="explorer",  color="#2196F3")
    axes[0].plot(steps, [r["raw_w_s"] for r in rows], label="survivor",  color="#F44336")
    axes[0].plot(steps, [r["raw_w_c"] for r in rows], label="craftsman", color="#4CAF50")
    axes[0].plot(steps, [r["raw_w_w"] for r in rows], label="warrior",   color="#9C27B0")
    axes[0].set_ylabel("Raw sigmoid")
    axes[0].set_title("Pre-normalisation activations (each in [0, 1] independently)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(steps, [r["norm_w_e"] for r in rows], label="explorer",  color="#2196F3")
    axes[1].plot(steps, [r["norm_w_s"] for r in rows], label="survivor",  color="#F44336")
    axes[1].plot(steps, [r["norm_w_c"] for r in rows], label="craftsman", color="#4CAF50")
    axes[1].plot(steps, [r["norm_w_w"] for r in rows], label="warrior",   color="#9C27B0")
    axes[1].set_ylabel("Normalised weight")
    axes[1].set_title("Post-normalisation weights (sum to 1)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(steps, health, label="health", color="#E91E63")
    axes[2].plot(steps, food,   label="food",   color="#FF9800")
    axes[2].plot(steps, drink,  label="drink",  color="#03A9F4")
    axes[2].plot(steps, energy, label="energy", color="#673AB7")
    axes[2].set_ylabel("Vital level")
    axes[2].set_xlabel("Step")
    axes[2].set_title("Vitals (drives Survivor's `need` signal)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(-0.05, 1.05)

    fig.suptitle(f"RIDGE blender diagnostic — survivor_baseline trajectory ({n_episodes} ep, sharpness={config.get('blend_sharpness', 1.0)})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out / f"blend_weights_diagnostic.{ext}"
        fig.savefig(path, dpi=120)
        logger.info("Plot saved to %s", path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Diagnose RIDGE blender weight normalisation")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/survivor_baseline_seed90598/best.pt")
    parser.add_argument("--config", type=str, default="configs/survivor.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="diagnostics/survivor_blend")
    args = parser.parse_args()

    from ridge.utils import load_default_config
    config = load_default_config(args.config)
    run_diagnostic(config, args.checkpoint, args.episodes, args.seed, args.out)


if __name__ == "__main__":
    main()
