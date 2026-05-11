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
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
    "legend.fontsize":   11,
    "figure.dpi":        150,
})

# Distinct linestyles so conditions separate even in greyscale print
_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
_MARKERS     = ["o", "s", "^", "D", "v", "P"]
_MARKER_EVERY = 20   # place a marker every N data-points


def _save_fig(fig: plt.Figure, out_path: str, dpi: int = 150) -> None:
    """Save figure as both PNG and PDF, creating parent dirs as needed."""
    stem = str(Path(out_path).with_suffix(""))
    png, pdf = stem + ".png", stem + ".pdf"
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    logger.info("Saved %s  +  %s", png, pdf)


CONDITION_COLOURS = {
    "ridge_adaptive":    "#4CAF50",
    "explorer_baseline": "#2196F3",
    "survivor_baseline": "#F44336",
    "craftsman_baseline": "#FF9800",
    "warrior_baseline":  "#9C27B0",
}
CONDITION_LABELS = {
    "ridge_adaptive":    "RIDGE",
    "explorer_baseline": "Explorer",
    "survivor_baseline": "Survivor",
    "craftsman_baseline": "Craftsman",
    "warrior_baseline":  "Warrior",
}


def launch_tensorboard(log_dir: str = "tensorboard_logs", port: int = 6006) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)]
    proc = subprocess.Popen(cmd)
    logger.info("TensorBoard launched on http://localhost:%d (logdir=%s)", port, log_dir)
    return proc


def _load_tb_scalars(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def _find_run_dirs(log_dir: str, condition: str) -> list[str]:
    base = Path(log_dir)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.iterdir() if p.is_dir() and p.name.startswith(condition))


def _mean_over_seeds(
    log_dir: str,
    condition: str,
    tag: str,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute mean ± std across seeds; returns (x, mean, std, n_seeds)."""
    run_dirs = _find_run_dirs(log_dir, condition)
    if not run_dirs:
        return np.array([]), np.array([]), np.array([]), 0

    all_steps, all_vals = [], []
    for rd in run_dirs:
        steps, vals = _load_tb_scalars(rd, tag)
        if len(steps):
            all_steps.append(steps)
            all_vals.append(vals)

    if not all_steps:
        return np.array([]), np.array([]), np.array([]), 0

    x_max    = min(s[-1] for s in all_steps)
    x_common = np.linspace(0, x_max, n_bins)
    interped = [np.interp(x_common, s, v) for s, v in zip(all_steps, all_vals)]
    arr      = np.stack(interped)
    return x_common, arr.mean(axis=0), arr.std(axis=0), len(arr)


def _add_condition_line(
    ax: plt.Axes,
    x: np.ndarray,
    mean_y: np.ndarray,
    std_y: np.ndarray,
    n_seeds: int,
    colour: str,
    label: str,
    style_idx: int,
) -> None:
    """Draw a mean line + shaded CI band with distinct linestyle and markers."""
    ls = _LINE_STYLES[style_idx % len(_LINE_STYLES)]
    mk = _MARKERS[style_idx % len(_MARKERS)]
    mark_every = max(1, len(x) // _MARKER_EVERY)

    full_label = f"{label} (n={n_seeds})" if n_seeds > 1 else label
    ax.plot(
        x, mean_y,
        label=full_label,
        color=colour,
        linewidth=2.5,
        linestyle=ls,
        marker=mk,
        markevery=mark_every,
        markersize=5,
        zorder=3,
    )
    ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=colour, alpha=0.18, zorder=2)


def _format_steps_axis(ax: plt.Axes) -> None:
    """Format x-axis as M (millions) for step counts."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1e3)}K"
    ))


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_achievement_coverage(
    log_dir: str,
    out_path: str = "results/achievement_coverage.png",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (cond, colour) in enumerate(CONDITION_COLOURS.items()):
        x, mean_y, std_y, n = _mean_over_seeds(log_dir, cond, "achievements/cumulative")
        if not len(x):
            continue
        _add_condition_line(ax, x, mean_y, std_y, n, colour,
                            CONDITION_LABELS.get(cond, cond), idx)

    ax.axhline(22, color="#555", linestyle=":", linewidth=1.2, label="Max (22)")
    _format_steps_axis(ax)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative Unique Achievements Unlocked")
    ax.set_title("Achievement Coverage — RIDGE vs Baselines")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_crafter_score(
    log_dir: str,
    out_path: str = "results/crafter_score.png",
) -> None:
    """Plot mean episode Crafter score over training steps."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (cond, colour) in enumerate(CONDITION_COLOURS.items()):
        x, mean_y, std_y, n = _mean_over_seeds(log_dir, cond, "episode/crafter_score")
        if not len(x):
            continue
        _add_condition_line(ax, x, mean_y, std_y, n, colour,
                            CONDITION_LABELS.get(cond, cond), idx)

    _format_steps_axis(ax)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Crafter Score (geometric mean success rate)")
    ax.set_title("Crafter Score over Training — RIDGE vs Baselines")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_training_stability(
    log_dir: str,
    out_path: str = "results/training_stability.png",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (cond, colour) in enumerate(CONDITION_COLOURS.items()):
        x, mean_y, std_y, n = _mean_over_seeds(log_dir, cond, "agent/value_loss")
        if not len(x):
            continue
        _add_condition_line(ax, x, mean_y, std_y, n, colour,
                            CONDITION_LABELS.get(cond, cond), idx)

    _format_steps_axis(ax)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Value Loss")
    ax.set_title("Training Stability — Value Loss over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_weight_trajectories(
    log_dir: str,
    run_name: str = "ridge_adaptive",
    out_path: str = "results/weight_trajectories.png",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    tag_styles = [
        ("weights/explorer",  "#4CAF50", "Explorer"),
        ("weights/survivor",  "#F44336", "Survivor"),
        ("weights/craftsman", "#FF9800", "Craftsman"),
        ("weights/warrior",   "#9C27B0", "Warrior"),
    ]
    for idx, (tag, colour, label) in enumerate(tag_styles):
        x, mean_y, std_y, n = _mean_over_seeds(log_dir, run_name, tag)
        if not len(x):
            continue
        _add_condition_line(ax, x, mean_y, std_y, n, colour, label, idx)

    _format_steps_axis(ax)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Persona Weight")
    ax.set_title("RIDGE — Persona Weight Trajectories")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_score_distribution(
    log_dir: str,
    out_path: str = "results/score_distribution.png",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    data, labels, colours = [], [], []
    for cond, colour in CONDITION_COLOURS.items():
        run_dirs = _find_run_dirs(log_dir, cond)
        scores = []
        for rd in run_dirs:
            _, vals = _load_tb_scalars(rd, "episode/crafter_score")
            if len(vals):
                scores.extend(vals[-200:].tolist())
        if scores:
            data.append(scores)
            labels.append(CONDITION_LABELS.get(cond, cond))
            colours.append(colour)

    if not data:
        plt.close(fig)
        return

    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        widths=0.55,
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(0)
    for i, (scores, colour) in enumerate(zip(data, colours), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(scores))
        ax.scatter(i + jitter, scores, color=colour, alpha=0.25, s=8, zorder=2)

    ax.set_ylabel("Crafter Score")
    ax.set_title("Score Distribution — Final 200 Episodes per Seed")
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_per_achievement_heatmap(
    log_dir: str,
    out_path: str = "results/achievement_heatmap.png",
) -> None:
    from ridge.game import ACHIEVEMENTS

    conditions = list(CONDITION_COLOURS.keys())
    labels     = [CONDITION_LABELS.get(c, c) for c in conditions]
    matrix     = np.zeros((len(ACHIEVEMENTS), len(conditions)), dtype=np.float32)

    for j, cond in enumerate(conditions):
        run_dirs = _find_run_dirs(log_dir, cond)
        for i, ach in enumerate(ACHIEVEMENTS):
            vals_all = []
            for rd in run_dirs:
                _, vals = _load_tb_scalars(rd, f"achievements/{ach}")
                if len(vals):
                    vals_all.append(vals[-100:].mean())
            if vals_all:
                matrix[i, j] = float(np.mean(vals_all))

    fig, ax = plt.subplots(figsize=(9, 13))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(ACHIEVEMENTS)))
    ax.set_yticklabels(
        [a.replace("_", " ").title() for a in ACHIEVEMENTS], fontsize=10
    )
    ax.set_title("Per-Achievement Success Rate (last 100 episodes)", pad=14)
    fig.colorbar(im, ax=ax, label="Success Rate", fraction=0.03, pad=0.02)

    # Annotate each cell with its value
    for i in range(len(ACHIEVEMENTS)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            text_col = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_col)

    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


# Sharpness ablation — colour scale: cool (soft) → warm (sharp)
SHARPNESS_CONDITIONS = {
    "ridge_adaptive": ("0.75 (default)", "#4CAF50"),
    "ridge_bs050":    ("0.5  (soft)",    "#80DEEA"),
    "ridge_bs100":    ("1.0",            "#FFF176"),
    "ridge_bs200":    ("2.0  (best)",    "#FF9800"),
    "ridge_bs400":    ("4.0  (hard)",    "#F44336"),
}


def plot_blend_sharpness(
    log_dir: str,
    out_path: str = "results/blend_sharpness.png",
    tag: str = "episode/crafter_score",
) -> None:
    """RQ3 — Crafter score vs blend_sharpness value."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (cond, (label, colour)) in enumerate(SHARPNESS_CONDITIONS.items()):
        x, mean_y, std_y, n = _mean_over_seeds(log_dir, cond, tag)
        if not len(x):
            continue
        _add_condition_line(ax, x, mean_y, std_y, n, colour, f"α={label}", idx)

    _format_steps_axis(ax)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Crafter Score (geometric mean success rate)")
    ax.set_title("RIDGE — Blend Sharpness Ablation (RQ3)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def generate_all_plots(log_dir: str = "tensorboard_logs", out_dir: str = "results") -> None:
    from rich.console import Console as _Console
    from rich.table import Table as _Table
    from rich import box as _box

    _c = _Console()

    # ── Data availability report ──────────────────────────────────────────────
    t = _Table(box=_box.SIMPLE, show_header=True, header_style="bold cyan")
    t.add_column("Condition",  style="white",      width=20)
    t.add_column("Run dirs",   style="bold",        width=10, justify="center")
    t.add_column("Status",     style="bold",        width=30)

    missing: list[str] = []
    for cond, colour in CONDITION_COLOURS.items():
        dirs = _find_run_dirs(log_dir, cond)
        label = CONDITION_LABELS.get(cond, cond)
        if dirs:
            t.add_row(label, str(len(dirs)), f"[green]✓ {dirs[0].split('/')[-1].split(chr(92))[-1]}[/]")
        else:
            t.add_row(label, "0", "[red]✗ No training data — run this condition first[/]")
            missing.append(label)

    _c.print(t)
    if missing:
        _c.print(
            f"  [bold yellow]⚠  {len(missing)}/5 conditions missing:[/] "
            f"{', '.join(missing)}\n"
            f"  [dim]Plots will show only the conditions above that have data.[/dim]\n"
        )

    plot_achievement_coverage(log_dir,    f"{out_dir}/achievement_coverage.png")
    plot_crafter_score(log_dir,           f"{out_dir}/crafter_score.png")
    plot_training_stability(log_dir,      f"{out_dir}/training_stability.png")
    plot_weight_trajectories(log_dir,     out_path=f"{out_dir}/weight_trajectories.png")
    plot_score_distribution(log_dir,      f"{out_dir}/score_distribution.png")
    plot_per_achievement_heatmap(log_dir, f"{out_dir}/achievement_heatmap.png")
    plot_blend_sharpness(log_dir,         f"{out_dir}/blend_sharpness.png")
    logger.info("All plots saved to %s/ (PNG + PDF)", out_dir)
