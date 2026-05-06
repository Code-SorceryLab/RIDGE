"""TensorBoard launcher and custom matplotlib comparison graphs."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CONDITION_COLOURS = {
    "ridge_adaptive": "#4CAF50",
    "explorer_baseline": "#2196F3",
    "survivor_baseline": "#F44336",
    "craftsman_baseline": "#FF9800",
}
CONDITION_LABELS = {
    "ridge_adaptive": "RIDGE",
    "explorer_baseline": "Explorer",
    "survivor_baseline": "Survivor",
    "craftsman_baseline": "Craftsman",
}


def launch_tensorboard(log_dir: str = "tensorboard_logs", port: int = 6006) -> subprocess.Popen:
    """Launch TensorBoard as a background subprocess.

    Args:
        log_dir: Directory containing TensorBoard event files.
        port: Port number for the TensorBoard server.

    Returns:
        Popen handle for the launched subprocess.
    """
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)]
    proc = subprocess.Popen(cmd)
    logger.info("TensorBoard launched on http://localhost:%d (logdir=%s)", port, log_dir)
    return proc


def _load_tb_scalars(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load scalar values for a given TensorBoard tag from event files.

    Args:
        log_dir: Path to a single TensorBoard run directory.
        tag: Scalar tag name (e.g. 'achievements/cumulative').

    Returns:
        Tuple of (steps, values) as float64 ndarrays.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def _find_run_dirs(log_dir: str, condition: str) -> list[str]:
    """Find all TensorBoard run directories for a given condition.

    Args:
        log_dir: Root log directory.
        condition: Condition prefix (e.g. 'ridge_adaptive').

    Returns:
        List of matching subdirectory paths.
    """
    base = Path(log_dir)
    return sorted(str(p) for p in base.iterdir() if p.is_dir() and p.name.startswith(condition))


def _mean_over_seeds(
    log_dir: str,
    condition: str,
    tag: str,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean ± std of a scalar tag across seeds via binned interpolation.

    Args:
        log_dir: Root log directory.
        condition: Condition name prefix.
        tag: TensorBoard scalar tag.
        n_bins: Number of x-axis bins for interpolation.

    Returns:
        Tuple of (x_common, mean_y, std_y) ndarrays.
    """
    run_dirs = _find_run_dirs(log_dir, condition)
    if not run_dirs:
        return np.array([]), np.array([]), np.array([])

    all_steps = []
    all_vals = []
    for rd in run_dirs:
        steps, vals = _load_tb_scalars(rd, tag)
        if len(steps):
            all_steps.append(steps)
            all_vals.append(vals)

    if not all_steps:
        return np.array([]), np.array([]), np.array([])

    x_max = min(s[-1] for s in all_steps)
    x_common = np.linspace(0, x_max, n_bins)
    interp_vals = [np.interp(x_common, s, v) for s, v in zip(all_steps, all_vals)]
    arr = np.stack(interp_vals)
    return x_common, arr.mean(axis=0), arr.std(axis=0)


def plot_achievement_coverage(
    log_dir: str,
    out_path: str = "results/achievement_coverage.png",
) -> None:
    """Plot cumulative unique achievements over training steps for all conditions.

    Args:
        log_dir: Root TensorBoard log directory.
        out_path: Output PNG file path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, colour in CONDITION_COLOURS.items():
        x, mean_y, std_y = _mean_over_seeds(log_dir, cond, "achievements/cumulative")
        if not len(x):
            continue
        label = CONDITION_LABELS.get(cond, cond)
        ax.plot(x, mean_y, label=label, color=colour, linewidth=2)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=colour, alpha=0.15)

    ax.axhline(22, color="gray", linestyle="--", linewidth=1, label="Max (22)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative Unique Achievements")
    ax.set_title("Achievement Coverage — RIDGE vs Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved achievement coverage plot to %s", out_path)


def plot_training_stability(
    log_dir: str,
    out_path: str = "results/training_stability.png",
) -> None:
    """Plot value loss over training steps for each condition.

    Args:
        log_dir: Root TensorBoard log directory.
        out_path: Output PNG file path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, colour in CONDITION_COLOURS.items():
        x, mean_y, std_y = _mean_over_seeds(log_dir, cond, "agent/value_loss")
        if not len(x):
            continue
        label = CONDITION_LABELS.get(cond, cond)
        ax.plot(x, mean_y, label=label, color=colour, linewidth=2)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=colour, alpha=0.15)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Value Loss")
    ax.set_title("Training Stability — Value Loss over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training stability plot to %s", out_path)


def plot_weight_trajectories(
    log_dir: str,
    run_name: str = "ridge_adaptive",
    out_path: str = "results/weight_trajectories.png",
) -> None:
    """Plot RIDGE persona weight trajectories over training for one run.

    Args:
        log_dir: Root TensorBoard log directory.
        run_name: Run directory prefix to use.
        out_path: Output PNG file path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    tag_colours = [
        ("weights/explorer", "#4CAF50", "Explorer"),
        ("weights/survivor", "#F44336", "Survivor"),
        ("weights/craftsman", "#FF9800", "Craftsman"),
    ]
    for tag, colour, label in tag_colours:
        x, mean_y, std_y = _mean_over_seeds(log_dir, run_name, tag)
        if not len(x):
            continue
        ax.plot(x, mean_y, label=label, color=colour, linewidth=2)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=colour, alpha=0.15)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Persona Weight")
    ax.set_title("RIDGE — Persona Weight Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved weight trajectories plot to %s", out_path)


def plot_score_distribution(
    log_dir: str,
    out_path: str = "results/score_distribution.png",
) -> None:
    """Box plots of episode score distributions across seeds.

    Args:
        log_dir: Root TensorBoard log directory.
        out_path: Output PNG file path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    data = []
    labels = []
    colours = []
    for cond, colour in CONDITION_COLOURS.items():
        run_dirs = _find_run_dirs(log_dir, cond)
        scores = []
        for rd in run_dirs:
            _, vals = _load_tb_scalars(rd, "episode/score")
            if len(vals):
                scores.extend(vals[-200:].tolist())
        if scores:
            data.append(scores)
            labels.append(CONDITION_LABELS.get(cond, cond))
            colours.append(colour)

    if data:
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

    ax.set_ylabel("Episode Score")
    ax.set_title("Score Distribution — Final 200 Episodes per Seed")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved score distribution plot to %s", out_path)


def plot_per_achievement_heatmap(
    log_dir: str,
    out_path: str = "results/achievement_heatmap.png",
) -> None:
    """Heatmap showing per-achievement success rates across conditions.

    Args:
        log_dir: Root TensorBoard log directory.
        out_path: Output PNG file path.
    """
    from ridge.game import ACHIEVEMENTS

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    conditions = list(CONDITION_COLOURS.keys())
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    matrix = np.zeros((len(ACHIEVEMENTS), len(conditions)), dtype=np.float32)

    for j, cond in enumerate(conditions):
        run_dirs = _find_run_dirs(log_dir, cond)
        for ach_i, ach in enumerate(ACHIEVEMENTS):
            tag = f"achievements/{ach}"
            vals_all = []
            for rd in run_dirs:
                _, vals = _load_tb_scalars(rd, tag)
                if len(vals):
                    vals_all.append(vals[-100:].mean())
            if vals_all:
                matrix[ach_i, j] = float(np.mean(vals_all))

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(len(ACHIEVEMENTS)))
    ax.set_yticklabels(ACHIEVEMENTS, fontsize=9)
    ax.set_title("Per-Achievement Success Rate")
    fig.colorbar(im, ax=ax, label="Success Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved achievement heatmap to %s", out_path)


def generate_all_plots(log_dir: str = "tensorboard_logs", out_dir: str = "results") -> None:
    """Generate the full comparison plot suite.

    Args:
        log_dir: Root TensorBoard log directory.
        out_dir: Output directory for PNG files.
    """
    plot_achievement_coverage(log_dir, f"{out_dir}/achievement_coverage.png")
    plot_training_stability(log_dir, f"{out_dir}/training_stability.png")
    plot_weight_trajectories(log_dir, out_path=f"{out_dir}/weight_trajectories.png")
    plot_score_distribution(log_dir, f"{out_dir}/score_distribution.png")
    plot_per_achievement_heatmap(log_dir, f"{out_dir}/achievement_heatmap.png")
    logger.info("All comparison plots saved to %s/", out_dir)
