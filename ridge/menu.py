"""Interactive CLI menu — main entry point for RIDGE."""

import logging
import random
from pathlib import Path
from typing import Any

from ridge.utils import load_default_config, setup_logging

logger = logging.getLogger(__name__)

_MENU = """
╔══════════════════════════════════════╗
║          RIDGE — Main Menu           ║
╠══════════════════════════════════════╣
║  1. Train RIDGE (adaptive blending)  ║
║  2. Train Explorer baseline          ║
║  3. Train Survivor baseline          ║
║  4. Train Craftsman baseline         ║
║  5. Train all conditions (sweep)     ║
║  6. Live Viewer (watch agent play)   ║
║  7. Launch TensorBoard               ║
║  8. Run comparison graphs            ║
║  9. Evaluate checkpoint              ║
║  0. Exit                             ║
╚══════════════════════════════════════╝
"""

_CONFIG_MAP = {
    "1": "configs/ridge_blend.yaml",
    "2": "configs/explorer.yaml",
    "3": "configs/survivor.yaml",
    "4": "configs/craftsman.yaml",
}


def _prompt_config(default_path: str) -> dict[str, Any]:
    """Prompt the user to select a config YAML, falling back to a default.

    Args:
        default_path: Path to the default config file.

    Returns:
        Loaded config dict.
    """
    user_path = input(f"Config path [{default_path}]: ").strip()
    path = user_path if user_path else default_path
    return load_default_config(path)


def _prompt_seed() -> int:
    """Prompt the user for a seed, defaulting to a random int.

    Returns:
        Seed integer.
    """
    default_seed = random.randint(0, 99999)
    raw = input(f"Seed [{default_seed}]: ").strip()
    try:
        return int(raw) if raw else default_seed
    except ValueError:
        print(f"Invalid seed, using {default_seed}")
        return default_seed


def _run_training(config_path: str) -> None:
    """Load config, prompt for seed, and launch training.

    Args:
        config_path: Path to the condition's YAML config.
    """
    from ridge.trainer import Trainer

    config = _prompt_config(config_path)
    seed = _prompt_seed()
    print(f"\nStarting training: {config.get('run_name', 'ridge')} | seed={seed}\n")
    trainer = Trainer(config, seed=seed)
    trainer.train()
    print("\nTraining complete.")
    _post_training_prompt(config)


def _post_training_prompt(config: dict[str, Any]) -> None:
    """Offer TensorBoard launch or comparison plots after training.

    Args:
        config: Completed run config dict.
    """
    choice = input("\nLaunch TensorBoard? [y/N]: ").strip().lower()
    if choice == "y":
        _do_launch_tensorboard(config.get("log_dir", "tensorboard_logs"))

    choice = input("Generate comparison graphs? [y/N]: ").strip().lower()
    if choice == "y":
        _do_comparison_graphs(config.get("log_dir", "tensorboard_logs"))


def _run_sweep() -> None:
    """Run all 4 conditions sequentially."""
    seed = _prompt_seed()
    for cond_key, config_path in _CONFIG_MAP.items():
        print(f"\n{'='*50}")
        print(f"Starting condition {cond_key}: {config_path}")
        config = _prompt_config(config_path)
        from ridge.trainer import Trainer
        trainer = Trainer(config, seed=seed)
        trainer.train()
    print("\nSweep complete.")


def _do_launch_tensorboard(log_dir: str = "tensorboard_logs") -> None:
    """Launch TensorBoard subprocess.

    Args:
        log_dir: TensorBoard log directory.
    """
    from viewer.dashboard import launch_tensorboard
    proc = launch_tensorboard(log_dir)
    print(f"TensorBoard running at http://localhost:6006  (PID {proc.pid})")
    print("Press Enter to return to menu...")
    input()


def _do_comparison_graphs(log_dir: str = "tensorboard_logs") -> None:
    """Generate and save all comparison plots.

    Args:
        log_dir: TensorBoard log directory.
    """
    from viewer.dashboard import generate_all_plots
    out_dir = input("Output directory [results]: ").strip() or "results"
    generate_all_plots(log_dir, out_dir)
    print(f"Plots saved to {out_dir}/")


def _do_live_viewer() -> None:
    """Launch the live viewer with a checkpoint."""
    from ridge.agent import PPOAgent
    from ridge.game import make_env
    from ridge.utils import get_device
    from viewer.live_viewer import LiveViewer

    config_path = input("Config path [configs/ridge_blend.yaml]: ").strip() or "configs/ridge_blend.yaml"
    config = load_default_config(config_path)

    ckpt_path = input("Checkpoint path: ").strip()
    if not ckpt_path or not Path(ckpt_path).exists():
        print("Checkpoint not found.")
        return

    device = get_device()
    env = make_env(config)
    agent = PPOAgent(config, num_actions=env.action_space.n, device=device)
    agent.load_checkpoint(ckpt_path)

    viewer = LiveViewer(render_fps=int(config.get("render_fps", 15)))
    viewer.run_replay(env, agent, config, ckpt_path)


def _do_evaluate() -> None:
    """Evaluate a checkpoint over N episodes."""
    config_path = input("Config path [configs/ridge_blend.yaml]: ").strip() or "configs/ridge_blend.yaml"
    config = load_default_config(config_path)
    ckpt_path = input("Checkpoint path: ").strip()
    n_eps = int(input("Number of episodes [10]: ").strip() or "10")
    seed = _prompt_seed()

    from scripts.evaluate import run_evaluation
    run_evaluation(config, ckpt_path, n_episodes=n_eps, seed=seed)


def run_menu() -> None:
    """Display the RIDGE main menu and dispatch user selections."""
    setup_logging()
    while True:
        print(_MENU)
        choice = input("Select option: ").strip()

        if choice == "1":
            _run_training("configs/ridge_blend.yaml")
        elif choice == "2":
            _run_training("configs/explorer.yaml")
        elif choice == "3":
            _run_training("configs/survivor.yaml")
        elif choice == "4":
            _run_training("configs/craftsman.yaml")
        elif choice == "5":
            _run_sweep()
        elif choice == "6":
            _do_live_viewer()
        elif choice == "7":
            _do_launch_tensorboard()
        elif choice == "8":
            _do_comparison_graphs()
        elif choice == "9":
            _do_evaluate()
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print("Invalid option. Please try again.")
