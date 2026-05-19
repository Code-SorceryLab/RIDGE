"""Shared constants and data loaders for RIDGE spider/radar charts.

Used by both viewer/dashboard.py (matplotlib export) and
viewer/post_training_dashboard.py (Plotly interactive).
"""

from pathlib import Path
from typing import Any

import numpy as np


ACHIEVEMENT_CATEGORIES: dict[str, list[str]] = {
    "Collection": [
        "collect_wood", "collect_stone", "collect_coal", "collect_iron",
        "collect_diamond", "collect_sapling",
    ],
    "Survival": [
        "collect_drink", "eat_cow", "eat_plant", "wake_up",
    ],
    "Crafting": [
        "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
        "place_table", "place_furnace", "place_stone", "place_plant",
    ],
    "Combat": [
        "defeat_zombie", "defeat_skeleton",
        "make_wood_sword", "make_stone_sword", "make_iron_sword",
    ],
}
ALL_ACHIEVEMENTS: list[str] = [a for axs in ACHIEVEMENT_CATEGORIES.values() for a in axs]

SPIDER_CONDITIONS: dict[str, tuple[str, str]] = {
    "ridge_adaptive":     ("RIDGE (main)", "#2E7D32"),
    "ridge_bs000":        ("RIDGE α=0.0",  "#80DEEA"),
    "ridge_bs050":        ("RIDGE α=0.5",  "#4DD0E1"),
    "ridge_bs100":        ("RIDGE α=1.0",  "#4CAF50"),
    "ridge_bs110":        ("RIDGE α=1.1",  "#8BC34A"),
    "ridge_bs120":        ("RIDGE α=1.2",  "#CDDC39"),
    "ridge_bs130":        ("RIDGE α=1.3",  "#FFEB3B"),
    "ridge_bs140":        ("RIDGE α=1.4",  "#FFC107"),
    "ridge_bs150":        ("RIDGE α=1.5",  "#FF9800"),
    "ridge_bs160":        ("RIDGE α=1.6",  "#FF5722"),
    "ridge_bs170":        ("RIDGE α=1.7",  "#E64A19"),
    "ridge_bs200":        ("RIDGE α=2.0",  "#B71C1C"),
    "explorer_baseline":  ("Explorer",     "#2196F3"),
    "survivor_baseline":  ("Survivor",     "#F44336"),
    "craftsman_baseline": ("Craftsman",    "#FFB74D"),
    "warrior_baseline":   ("Warrior",      "#9C27B0"),
    "all_ones_baseline":  ("All-Ones",     "#607D8B"),
}
SHARPNESS_KEYS = (
    "ridge_adaptive",
    "ridge_bs000", "ridge_bs050",
    "ridge_bs100", "ridge_bs110", "ridge_bs120", "ridge_bs130",
    "ridge_bs140", "ridge_bs150", "ridge_bs160", "ridge_bs170",
    "ridge_bs200",
)
FIXED_BASELINE_KEYS = (
    "explorer_baseline", "survivor_baseline", "craftsman_baseline",
    "warrior_baseline", "all_ones_baseline",
)

TRAINING_METRIC_AXES: list[tuple[str, str, str]] = [
    ("episode/crafter_score",   "Crafter Score",     "global_max"),
    ("achievements/cumulative", "Achievements / 22", "fixed:22"),
    ("agent/value_loss",        "Value Stability",   "invert_minmax"),
    ("agent/entropy",           "Exploration (H)",   "global_max"),
]


# ─── Data loading (no Streamlit dependency — caller adds caching as needed) ───

def find_run_dirs(log_dir: str, prefix: str) -> list[str]:
    """Return run dirs whose name is exactly `prefix` or `prefix` followed by
    `_seed…`. The trailing-boundary check prevents short prefixes
    (e.g. `ridge_bs10`) from accidentally matching longer condition runs
    (e.g. `ridge_bs100_seed42_…`) and merging unrelated sweeps.
    """
    base = Path(log_dir)
    if not base.exists():
        return []
    return sorted(
        str(p) for p in base.iterdir()
        if p.is_dir() and (p.name == prefix or p.name.startswith(prefix + "_seed"))
    )


def load_condition(log_dir: str, prefix: str) -> dict[str, Any] | None:
    """One-pass mean-over-seeds load of every spider-relevant scalar."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    dirs = find_run_dirs(log_dir, prefix)
    if not dirs:
        return None

    wanted: set[str] = set()
    wanted.update(f"achievements/{a}" for a in ALL_ACHIEVEMENTS)
    wanted.add("achievements/cumulative")
    wanted.update(t[0] for t in TRAINING_METRIC_AXES)
    wanted.update(f"weights/{p}" for p in ("explorer", "survivor", "craftsman", "warrior"))

    buckets: dict[str, list[float]] = {}
    max_steps: list[int] = []
    for d in dirs:
        try:
            ea = EventAccumulator(d, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        avail = set(ea.Tags().get("scalars", []))
        for tag in wanted & avail:
            evs = ea.Scalars(tag)
            if evs:
                buckets.setdefault(tag, []).append(
                    float(np.mean([e.value for e in evs[-100:]]))
                )
        for tag in ("episode/crafter_score", "achievements/cumulative", "agent/value_loss"):
            if tag in avail:
                evs = ea.Scalars(tag)
                if evs:
                    max_steps.append(int(evs[-1].step))
                    break

    out: dict[str, Any] = {
        "achievements": {}, "metrics": {}, "weights": {},
        "n_seeds": len(dirs), "max_step": max(max_steps) if max_steps else 0,
    }
    for tag, vals in buckets.items():
        m = float(np.mean(vals))
        if tag.startswith("achievements/"):
            a = tag.split("/", 1)[1]
            if a in ALL_ACHIEVEMENTS:
                out["achievements"][a] = m
            else:
                out["metrics"][tag] = m
        elif tag.startswith("weights/"):
            out["weights"][tag.split("/", 1)[1]] = m
        else:
            out["metrics"][tag] = m

    # Override episode/crafter_score with the official Hafner 2022 formula
    # computed offline from achievement rates. This keeps OLD runs (logged
    # under a non-Hafner formula) and NEW runs on a single consistent scale.
    rates_pct = np.array(
        [100.0 * out["achievements"].get(a, 0.0) for a in ALL_ACHIEVEMENTS]
    )
    if rates_pct.size:
        out["metrics"]["episode/crafter_score"] = float(
            np.exp(np.mean(np.log1p(rates_pct))) - 1.0
        )
    return out


def available_conditions(log_dir: str) -> list[str]:
    return [c for c in SPIDER_CONDITIONS if find_run_dirs(log_dir, c)]


# ─── Series computation (used by both renderers) ──────────────────────────────

def _persona_one_hot(condition: str) -> list[float]:
    return {
        "explorer_baseline":  [1.0, 0.0, 0.0, 0.0],
        "survivor_baseline":  [0.0, 1.0, 0.0, 0.0],
        "craftsman_baseline": [0.0, 0.0, 1.0, 0.0],
        "warrior_baseline":   [0.0, 0.0, 0.0, 1.0],
    }.get(condition, [0.0, 0.0, 0.0, 0.0])


def compute_view_data(
    view: str,
    shown: list[str],
    data_by_cond: dict[str, dict],
) -> tuple[list[str], list[tuple[str, list[float], str]]]:
    """Return (axis_labels, [(display_label, values, colour), ...]) for one view."""
    series: list[tuple[str, list[float], str]] = []

    if view == "categories":
        axis_labels = list(ACHIEVEMENT_CATEGORIES.keys())
        for c in shown:
            d = data_by_cond.get(c)
            if d is None:
                continue
            cat_vals = [
                float(np.mean([d["achievements"].get(a, 0.0) for a in axs]))
                for axs in ACHIEVEMENT_CATEGORIES.values()
            ]
            series.append((SPIDER_CONDITIONS[c][0], cat_vals, SPIDER_CONDITIONS[c][1]))

    elif view == "metrics":
        raw = {tag: {c: data_by_cond.get(c, {}).get("metrics", {}).get(tag) for c in shown}
               for tag, _, _ in TRAINING_METRIC_AXES}
        norm: dict[str, dict[str, float]] = {}
        for tag, _, strategy in TRAINING_METRIC_AXES:
            vals = [v for v in raw[tag].values() if v is not None]
            if not vals:
                norm[tag] = {c: 0.0 for c in shown}
                continue
            if strategy == "fixed:22":
                norm[tag] = {c: float((raw[tag][c] or 0.0) / 22.0) for c in shown}
            elif strategy == "invert_minmax":
                lo, hi = min(vals), max(vals)
                rng = hi - lo if hi > lo else 1.0
                norm[tag] = {c: (float(1.0 - (raw[tag][c] - lo) / rng)
                                  if raw[tag][c] is not None else 0.0) for c in shown}
            else:
                hi = max(vals) if max(vals) > 0 else 1.0
                norm[tag] = {c: (float(raw[tag][c] / hi)
                                  if raw[tag][c] is not None else 0.0) for c in shown}
        axis_labels = [a[1] for a in TRAINING_METRIC_AXES]
        for c in shown:
            if c not in data_by_cond:
                continue
            vals = [norm[t[0]][c] for t in TRAINING_METRIC_AXES]
            series.append((SPIDER_CONDITIONS[c][0], vals, SPIDER_CONDITIONS[c][1]))

    elif view == "weights":
        personas = ["explorer", "survivor", "craftsman", "warrior"]
        axis_labels = [p.capitalize() for p in personas]
        for c in shown:
            d = data_by_cond.get(c)
            if d is None:
                continue
            w = d.get("weights", {})
            vals = ([float(w.get(p, 0.0)) for p in personas]
                    if w else _persona_one_hot(c))
            series.append((SPIDER_CONDITIONS[c][0], vals, SPIDER_CONDITIONS[c][1]))

    else:  # all_22
        axis_labels = [a.replace("_", " ").title() for a in ALL_ACHIEVEMENTS]
        for c in shown:
            d = data_by_cond.get(c)
            if d is None:
                continue
            vals = [d["achievements"].get(a, 0.0) for a in ALL_ACHIEVEMENTS]
            series.append((SPIDER_CONDITIONS[c][0], vals, SPIDER_CONDITIONS[c][1]))

    return axis_labels, series


VIEW_META: list[tuple[str, str, str]] = [
    ("categories", "Achievement categories",
     "Mean success rate within each Crafter category (0 → 1)"),
    ("metrics",    "Training metrics",
     "Per-axis normalised: 1.0 = best on each axis (value-loss inverted)"),
    ("weights",    "Mean persona weights",
     "Mean blending weights across training (sum to 1)"),
    ("all_22",     "All 22 achievements",
     "Per-achievement success rate — all 22 Crafter goals"),
]
