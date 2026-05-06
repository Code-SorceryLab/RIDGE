"""Entry point: launch the RIDGE menu or start training directly via args."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIDGE training entry point")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (skips menu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--live", action="store_true", help="Enable live viewer during training")
    return parser.parse_args()


def main() -> None:
    """Parse args and either show the interactive menu or start training directly."""
    args = _parse_args()

    if args.config is None:
        from ridge.menu import run_menu
        run_menu()
        return

    from ridge.trainer import Trainer
    from ridge.utils import load_default_config, setup_logging

    setup_logging()
    config = load_default_config(args.config)

    if args.live:
        config["live_view"] = True

    trainer = Trainer(config, seed=args.seed)
    trainer.train()


if __name__ == "__main__":
    main()
