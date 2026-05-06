"""Load a checkpoint and evaluate the agent over N episodes, exporting metrics."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def run_evaluation(
    config: dict[str, Any],
    checkpoint_path: str,
    n_episodes: int = 10,
    seed: int = 42,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate a trained agent and return aggregate metrics.

    Args:
        config: Project config dict.
        checkpoint_path: Path to the .pt checkpoint file.
        n_episodes: Number of evaluation episodes to run.
        seed: Seed for the evaluation environment.
        out_path: Optional JSON path to write metrics to.

    Returns:
        Dict with mean/std of total_reward, achievement_count, and episode_length.
    """
    from ridge.agent import PPOAgent
    from ridge.game import make_env
    from ridge.rewards import compute_blended_reward
    from ridge.utils import get_device, set_seeds, setup_logging

    setup_logging()
    set_seeds(seed)
    device = get_device()

    env = make_env(config, seed=seed)
    agent = PPOAgent(config, num_actions=env.action_space.n, device=device)
    agent.load_checkpoint(checkpoint_path)
    agent.network.eval()

    all_rewards: list[float] = []
    all_achievements: list[int] = []
    all_lengths: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            state_vec = env.extract_state_vector(info)
            blended, weights, _ = compute_blended_reward(info, state_vec, config)
            action, _, _, _ = agent.select_action(obs, weights)
            obs, _, terminated, truncated, info = env.step(action)
            env.update_episode_stats(blended, info, weights)
            done = terminated or truncated
            ep_reward += blended
            steps += 1

        ep_stats = env.get_episode_stats().to_dict()
        all_rewards.append(ep_stats["total_reward"])
        all_achievements.append(ep_stats["achievement_count"])
        all_lengths.append(ep_stats["steps"])
        logger.info("Episode %d/%d — reward=%.2f achievements=%d steps=%d",
                    ep + 1, n_episodes, ep_stats["total_reward"],
                    ep_stats["achievement_count"], ep_stats["steps"])

    env.close()

    results = {
        "n_episodes": n_episodes,
        "checkpoint": checkpoint_path,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_achievements": float(np.mean(all_achievements)),
        "std_achievements": float(np.std(all_achievements)),
        "mean_length": float(np.mean(all_lengths)),
        "std_length": float(np.std(all_lengths)),
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results written to %s", out_path)

    logger.info(
        "Eval complete — mean_reward=%.2f  mean_achievements=%.1f",
        results["mean_reward"],
        results["mean_achievements"],
    )
    return results


def main() -> None:
    """CLI entry point for evaluate.py."""
    parser = argparse.ArgumentParser(description="Evaluate a RIDGE checkpoint")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    from ridge.utils import load_default_config
    config = load_default_config(args.config)
    run_evaluation(config, args.checkpoint, n_episodes=args.episodes, seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
