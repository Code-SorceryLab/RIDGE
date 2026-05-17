"""Run a list of configs over a list of seeds, sequentially.

Designed for unattended overnight / multi-day runs to get cross-seed CIs on
the headline comparison. Each (config, seed) pair produces:

    checkpoints/<run_name>_seed<seed>/
    tensorboard_logs/<run_name>_seed<seed>/

These accumulate alongside existing single-seed runs; nothing is overwritten
(new seed → new directory). `dashboard.py:_mean_over_seeds` picks them up
automatically and renders mean ± std bands.

Examples:

  # The "paper-headline" set: 3 critical configs at 2 new seeds each = 6 runs.
  python scripts/run_multi_seed.py \
      --configs configs/ridge_sharp100.yaml \
                configs/ridge_sharp150.yaml \
                configs/craftsman.yaml \
      --seeds 11111 22222

  # Single condition at 3 random seeds.
  python scripts/run_multi_seed.py \
      --configs configs/ridge_sharp150.yaml \
      --random 3

  # Resume tolerance: if a (config, seed) directory already has a 1M-step
  # checkpoint, the trainer will resume from it (auto_resume defaults to true).
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple (config × seed) training runs sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--configs", nargs="+", required=True,
                         help="YAML config paths to run")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--seeds", nargs="+", type=int,
                       help="Explicit seeds to use (e.g. --seeds 11111 22222 33333)")
    grp.add_argument("--random", type=int, dest="n_random",
                       help="Generate N random seeds in [10000, 99999]")
    args = parser.parse_args()

    from ridge.trainer import Trainer
    from ridge.utils import load_default_config, setup_logging

    setup_logging()

    seeds: list[int] = (
        list(args.seeds) if args.seeds is not None
        else [random.randint(10000, 99999) for _ in range(args.n_random)]
    )
    total = len(args.configs) * len(seeds)

    print()
    print("=" * 64)
    print(f"  Multi-seed run — {len(args.configs)} configs × {len(seeds)} seeds = {total} runs")
    print(f"  Configs: {', '.join(args.configs)}")
    print(f"  Seeds:   {', '.join(str(s) for s in seeds)}")
    print("=" * 64)

    t_start = time.time()
    completed = 0
    for config_path in args.configs:
        for seed in seeds:
            completed += 1
            elapsed_min = (time.time() - t_start) / 60.0
            print()
            print("-" * 64)
            print(f"  [{completed}/{total}] {config_path}  seed={seed}  "
                   f"(elapsed {elapsed_min:.1f} min)")
            print("-" * 64)
            config = load_default_config(config_path)
            Trainer(config, seed=seed).train()
            print(f"  ✓ run {completed}/{total} complete "
                   f"(run_name={config.get('run_name', '?')}, seed={seed})")

    total_min = (time.time() - t_start) / 60.0
    print()
    print("=" * 64)
    print(f"  All {total} runs complete in {total_min:.1f} min "
           f"({total_min / 60.0:.1f} h)")
    print("=" * 64)


if __name__ == "__main__":
    main()
