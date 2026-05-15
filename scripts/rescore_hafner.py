"""Recompute Crafter score using the official Hafner 2022 formula on existing TB logs.

No retraining or re-evaluation. Reads the per-achievement success rates that
were already logged during training, applies:

    score = exp(mean(log(1 + 100·rate))) − 1

and prints a ranked comparison table. Saves a JSON for downstream use.

Usage:
    python scripts/rescore_hafner.py [--log-dir tensorboard_logs] [--out results/hafner_scores.json]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from viewer.spider_data import ALL_ACHIEVEMENTS, SPIDER_CONDITIONS, find_run_dirs


def hafner_score(rates_pct: np.ndarray) -> float:
    """Official Hafner 2022 Crafter score: geometric mean of 22 achievement %'s with +1 offset."""
    return float(np.exp(np.mean(np.log1p(rates_pct))) - 1.0)


def load_run_data(run_dir: str, tail: int = 100) -> dict[str, Any] | None:
    """Pull per-achievement success rates + old crafter_score for one run dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    try:
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
    except Exception:
        return None
    available = set(ea.Tags().get("scalars", []))

    rates: dict[str, float] = {}
    for ach in ALL_ACHIEVEMENTS:
        tag = f"achievements/{ach}"
        if tag in available:
            evs = ea.Scalars(tag)
            if evs:
                rates[ach] = float(np.mean([e.value for e in evs[-tail:]]))

    old_score = None
    if "episode/crafter_score" in available:
        evs = ea.Scalars("episode/crafter_score")
        if evs:
            old_score = float(np.mean([e.value for e in evs[-tail:]]))

    cum_ach = None
    if "achievements/cumulative" in available:
        evs = ea.Scalars("achievements/cumulative")
        if evs:
            cum_ach = float(evs[-1].value)

    return {"rates": rates, "old_score": old_score, "cum_achievements": cum_ach}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore existing runs with Hafner 2022 Crafter formula")
    parser.add_argument("--log-dir", default="tensorboard_logs")
    parser.add_argument("--out", default="results/hafner_scores.json")
    parser.add_argument("--tail", type=int, default=100,
                         help="Mean of last N episodes for each rate (default 100)")
    args = parser.parse_args()

    print()
    print(f"  {'Condition':<22} {'n':>3}  {'Old (×100)':>10} {'Hafner %':>10} {'Δ':>8}  {'Cum/22':>7}")
    print("  " + "-" * 70)

    all_results: dict[str, Any] = {}
    for cond, (label, _) in SPIDER_CONDITIONS.items():
        dirs = find_run_dirs(args.log_dir, cond)
        if not dirs:
            continue
        hafner_scores: list[float] = []
        old_scores: list[float] = []
        cum_achs: list[float] = []
        for d in dirs:
            data = load_run_data(d, tail=args.tail)
            if data is None or not data["rates"]:
                continue
            rates_pct = np.array(
                [100.0 * data["rates"].get(ach, 0.0) for ach in ALL_ACHIEVEMENTS]
            )
            hafner_scores.append(hafner_score(rates_pct))
            if data["old_score"] is not None:
                old_scores.append(data["old_score"])
            if data["cum_achievements"] is not None:
                cum_achs.append(data["cum_achievements"])

        if not hafner_scores:
            continue

        old_mean_pct  = (float(np.mean(old_scores)) * 100.0) if old_scores else float("nan")
        new_mean      = float(np.mean(hafner_scores))
        new_std       = float(np.std(hafner_scores)) if len(hafner_scores) > 1 else 0.0
        delta         = new_mean - (old_mean_pct if not np.isnan(old_mean_pct) else 0.0)
        cum_mean      = float(np.mean(cum_achs)) if cum_achs else float("nan")
        n             = len(hafner_scores)
        std_str       = f"±{new_std:4.2f}" if n > 1 else ""
        print(f"  {label:<22} {n:>3}  {old_mean_pct:>9.2f}% {new_mean:>8.2f}% {std_str:<5} "
               f"{delta:>+7.2f}  {cum_mean:>6.1f}")

        all_results[cond] = {
            "label":               label,
            "n_seeds":             n,
            "old_score_fraction":  float(np.mean(old_scores)) if old_scores else None,
            "old_score_pct":       old_mean_pct,
            "hafner_score_pct":    new_mean,
            "hafner_score_std":    new_std,
            "delta_pct":           delta,
            "mean_cum_achievements": cum_mean,
            "per_seed_hafner_pct": hafner_scores,
        }

    # Ranked summary
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["hafner_score_pct"], reverse=True)
    print()
    print("  Ranked by Hafner score:")
    print("  " + "-" * 40)
    for rank, (_, r) in enumerate(ranked, 1):
        std_suffix = f" ±{r['hafner_score_std']:.2f}" if r["n_seeds"] > 1 else ""
        print(f"    {rank}. {r['label']:<22} {r['hafner_score_pct']:>6.2f}%{std_suffix}")

    # Reference comparison
    print()
    print("  Reference: Hafner 2022 (DreamerV3, 1M steps) = 21.79%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
