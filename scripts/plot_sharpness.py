"""
plot_sharpness.py

Reads results/sharpness_sweep/summary.json and generates four graphs:

  1. Mean episode reward vs blend_sharpness
  2. Mean achievements per episode vs blend_sharpness
  3. Per-head value loss (final 100 updates) vs blend_sharpness  ← RQ3 stability
  4. Mean persona weight vs blend_sharpness  ← shows blending behaviour

Usage:
  python scripts/plot_sharpness.py
  python scripts/plot_sharpness.py --results results/sharpness_sweep/summary.json --out results/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PERSONA_NAMES  = ["explorer", "survivor", "craftsman", "warrior"]
PERSONA_COLORS = {
    "explorer":  "#4C9BE8",
    "survivor":  "#E87A4C",
    "craftsman": "#4CE87A",
    "warrior":   "#C44CE8",
}

FIG_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "text.color":       "#c9d1d9",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "lines.linewidth":  2.0,
    "font.size":        11,
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return sorted(data, key=lambda r: r["blend_sharpness"])


def _xs(results: list[dict]) -> np.ndarray:
    return np.array([r["blend_sharpness"] for r in results])


def plot_reward(ax: plt.Axes, results: list[dict]) -> None:
    xs    = _xs(results)
    means = np.array([r["mean_reward"] for r in results])
    stds  = np.array([r["std_reward"]  for r in results])
    ax.plot(xs, means, color="#4C9BE8", marker="o")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="#4C9BE8")
    ax.set_title("Mean Episode Reward")
    ax.set_xlabel("blend_sharpness")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.axvline(1.0, color="#8b949e", linestyle="--", linewidth=1, alpha=0.6, label="default (1.0)")
    ax.legend(fontsize=9)


def plot_achievements(ax: plt.Axes, results: list[dict]) -> None:
    xs    = _xs(results)
    means = np.array([r["mean_achievements"] for r in results])
    stds  = np.array([r["std_achievements"]  for r in results])
    ax.plot(xs, means, color="#4CE87A", marker="o")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="#4CE87A")
    ax.set_title("Mean Achievements per Episode")
    ax.set_xlabel("blend_sharpness")
    ax.set_ylabel("Achievements unlocked")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True)
    ax.axvline(1.0, color="#8b949e", linestyle="--", linewidth=1, alpha=0.6, label="default (1.0)")
    ax.legend(fontsize=9)


def plot_value_losses(ax: plt.Axes, results: list[dict]) -> None:
    """Per-head value loss (final 100 updates) — RQ3 stability signal.

    Higher / more divergent losses at high sharpness = switching-stability dilemma.
    """
    xs = _xs(results)
    for persona in PERSONA_NAMES:
        ys = np.array([
            r["value_loss_summary"][persona]["final_100_mean"]
            for r in results
        ])
        ax.plot(xs, ys, color=PERSONA_COLORS[persona], marker="o", label=persona)
    ax.set_title("Per-head Value Loss (final 100 updates) — RQ3")
    ax.set_xlabel("blend_sharpness")
    ax.set_ylabel("MSE value loss")
    ax.grid(True)
    ax.axvline(1.0, color="#8b949e", linestyle="--", linewidth=1, alpha=0.6, label="default")
    ax.legend(fontsize=9)


def plot_weights(ax: plt.Axes, results: list[dict]) -> None:
    """Mean persona weight across all episodes — shows blending behaviour."""
    xs = _xs(results)
    bottoms = np.zeros(len(xs))
    for persona in PERSONA_NAMES:
        ys = np.array([
            r["weight_summary"].get(persona, {}).get("mean_weight", 0.25)
            for r in results
        ])
        ax.bar(xs, ys, bottom=bottoms, label=persona,
               color=PERSONA_COLORS[persona], alpha=0.85, width=0.08)
        bottoms += ys
    ax.set_title("Mean Persona Weight vs Sharpness")
    ax.set_xlabel("blend_sharpness")
    ax.set_ylabel("Mean weight (stacked)")
    ax.set_ylim(0, 1.05)
    ax.axvline(1.0, color="#8b949e", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y")


def plot_achievement_heatmap(results: list[dict], out_dir: Path) -> None:
    """Unlock rate per achievement per sharpness — separate figure."""
    xs = _xs(results)
    achievements = [
        a for a in [
            "collect_wood", "collect_stone", "collect_coal", "collect_iron",
            "collect_diamond", "make_wood_pickaxe", "make_stone_pickaxe",
            "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
            "make_iron_sword", "defeat_zombie", "defeat_skeleton",
            "eat_plant", "eat_cow", "wake_up", "place_table", "place_furnace",
        ]
    ]
    matrix = np.array([
        [r["achievement_unlock_rates"].get(a, 0.0) for r in results]
        for a in achievements
    ])  # (n_achievements, n_sharpness)

    with plt.style.context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(xs) * 1.2), len(achievements) * 0.45 + 1.5))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:.2f}" for x in xs], fontsize=9)
        ax.set_yticks(range(len(achievements)))
        ax.set_yticklabels(achievements, fontsize=9)
        ax.set_xlabel("blend_sharpness")
        ax.set_title("Achievement Unlock Rate (%) by Sharpness")
        plt.colorbar(im, ax=ax, label="Unlock rate (%)")
        fig.tight_layout()
        path = out_dir / "sharpness_achievement_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot blend_sharpness sweep results")
    parser.add_argument("--results", type=str,
                        default="results/sharpness_sweep/summary.json")
    parser.add_argument("--out",     type=str, default="results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)
    print(f"Loaded {len(results)} conditions: sharpness = {[r['blend_sharpness'] for r in results]}")

    # ── 2×2 main figure ──────────────────────────────────────────────────────
    with plt.style.context(FIG_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("RIDGE — blend_sharpness Sweep (RQ3)", fontsize=14, y=1.01)

        plot_reward      (axes[0, 0], results)
        plot_achievements(axes[0, 1], results)
        plot_value_losses(axes[1, 0], results)
        plot_weights     (axes[1, 1], results)

        fig.tight_layout()
        path = out_dir / "sharpness_sweep.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Achievement heatmap — separate figure ────────────────────────────────
    plot_achievement_heatmap(results, out_dir)

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
