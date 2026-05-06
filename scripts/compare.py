"""Generate multi-run comparison plots and summary tables."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """CLI entry point: generate all comparison plots from a TensorBoard log dir."""
    parser = argparse.ArgumentParser(description="Generate RIDGE comparison plots")
    parser.add_argument("--logdir", type=str, default="tensorboard_logs", help="TensorBoard log directory")
    parser.add_argument("--out", type=str, default="results", help="Output directory for plots")
    args = parser.parse_args()

    from ridge.utils import setup_logging
    from viewer.dashboard import generate_all_plots

    setup_logging()
    generate_all_plots(log_dir=args.logdir, out_dir=args.out)
    print(f"All comparison plots written to {args.out}/")


if __name__ == "__main__":
    main()
