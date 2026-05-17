"""RIDGE Live Training Monitor — Streamlit dashboard.

Launch with:
    streamlit run viewer/streamlit_dashboard.py

Reads training_live/metrics.json and training_live/frame.npy written by the
trainer every rollout when live_dashboard: true is set in the config.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import streamlit as st

METRICS_FILE = Path("training_live/metrics.json")
FRAME_FILE   = Path("training_live/frame.npy")
REFRESH_S    = 2

PERSONA_COLOURS = {
    "explorer":  "#4CAF50",
    "survivor":  "#F44336",
    "craftsman": "#FF9800",
    "warrior":   "#9C27B0",
}


def _load_metrics() -> dict | None:
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_frame() -> np.ndarray | None:
    try:
        arr = np.load(FRAME_FILE)
        return arr.astype(np.uint8)
    except (FileNotFoundError, ValueError, OSError):
        return None


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RIDGE Monitor",
    layout="wide",
    page_icon="🎮",
    initial_sidebar_state="expanded",
)

# ── Sidebar — System stats ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## System Monitor")

    cpu_pcts = psutil.cpu_percent(percpu=True, interval=0.1)
    for i, pct in enumerate(cpu_pcts):
        colour = "normal"
        st.progress(min(1.0, pct / 100), text=f"Core {i}: {pct:.0f}%")

    mem = psutil.virtual_memory()
    used_gb  = mem.used  / 1e9
    total_gb = mem.total / 1e9
    st.progress(mem.percent / 100, text=f"RAM: {used_gb:.1f} / {total_gb:.1f} GB")

    st.divider()
    st.caption(f"Auto-refresh every {REFRESH_S}s")
    st.caption("Set `live_dashboard: true` in your config to enable data streaming.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎮 RIDGE — Live Training Monitor")

metrics = _load_metrics()
frame   = _load_frame()

if metrics is None:
    st.info(
        "⏳ **Waiting for training to start.**\n\n"
        "Make sure `live_dashboard: true` is set in your config, then start training."
    )
    time.sleep(REFRESH_S)
    st.rerun()

# ── Top KPI row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Step",         f"{metrics['step']:,}")
c2.metric("Episode",      f"{metrics['episode']:,}")
c3.metric("FPS",          f"{metrics.get('fps', 0):.0f}")
c4.metric("Crafter Score",f"{metrics.get('score', 0):.4f}")
c5.metric("Achievements", f"{metrics.get('achievements', 0)} / 22")

st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# Left: game frame + vitals
with left:
    st.subheader("Game Frame")
    if frame is not None:
        st.image(frame, use_container_width=True)
    else:
        st.caption("No frame received yet.")

    st.subheader("Vitals")
    vitals = metrics.get("vitals", {})
    for label, key in [("Health", "health"), ("Food", "food"),
                        ("Drink", "drink"),  ("Energy", "energy")]:
        val = vitals.get(key, 9)
        pct = val / 9.0
        st.progress(pct, text=f"{label}: {val} / 9")

# Right: active persona banner + weights + reward breakdown
with right:
    weights = metrics.get("weights", [0.25, 0.25, 0.25, 0.25])
    _persona_names = ["Explorer", "Survivor", "Craftsman", "Warrior"]
    dominant_idx = int(max(range(len(weights)), key=lambda i: weights[i]))
    dominant_name = _persona_names[dominant_idx]
    dominant_colour = PERSONA_COLOURS[dominant_name.lower()]
    st.markdown(
        f"<div style='background:{dominant_colour}22;border-left:4px solid {dominant_colour};"
        f"padding:10px 16px;border-radius:6px;margin-bottom:12px'>"
        f"<span style='font-size:0.8em;color:{dominant_colour};font-weight:600;"
        f"letter-spacing:0.08em;text-transform:uppercase'>Active Persona</span><br>"
        f"<span style='font-size:1.6em;font-weight:700;color:{dominant_colour}'>"
        f"{dominant_name}</span>"
        f"&nbsp;<span style='font-size:1em;color:#aaa'>{weights[dominant_idx]:.3f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Persona Weights")
    for name, w in zip(_persona_names, weights):
        colour = PERSONA_COLOURS[name.lower()]
        st.progress(float(w), text=f"{name}: {w:.3f}")

    st.subheader("Episode Reward Breakdown")
    rewards = metrics.get("persona_rewards", {})
    max_r = max(abs(v) for v in rewards.values()) if rewards else 1.0
    if max_r < 1e-6:
        max_r = 1.0
    for name, key in [("Explorer", "explorer"), ("Survivor", "survivor"),
                       ("Craftsman", "craftsman"), ("Warrior", "warrior")]:
        r = rewards.get(key, 0.0)
        bar_val = min(1.0, max(0.0, r / max_r))
        st.progress(bar_val, text=f"{name}: {r:.2f}")

st.divider()

# ── Score history chart ───────────────────────────────────────────────────────
st.subheader("Crafter Score History (last 100 updates)")

if "score_history" not in st.session_state:
    st.session_state.score_history = []
if "step_history" not in st.session_state:
    st.session_state.step_history = []

st.session_state.score_history.append(metrics.get("score", 0.0))
st.session_state.step_history.append(metrics.get("step", 0))

if len(st.session_state.score_history) > 100:
    st.session_state.score_history.pop(0)
    st.session_state.step_history.pop(0)

if len(st.session_state.score_history) > 1:
    chart_df = pd.DataFrame(
        {"Crafter Score": st.session_state.score_history},
        index=st.session_state.step_history,
    )
    st.line_chart(chart_df, color="#4CAF50")

st.divider()

# ── Achievement Map ───────────────────────────────────────────────────────────
#
# Organised by Crafter tech-tree tier.  All 22 achievements are reachable via
# the 17-action space; each has at least one persona reward bonus in rewards.py.
#
# Tier 1 — no prerequisites (basic interaction / exploration)
# Tier 2 — requires wood / crafting table
# Tier 3 — requires stone tools / furnace chain
# Tier 4 — requires iron tools (deep tech tree)
# ─────────────────────────────────────────────────────────────────────────────

_TECH_TREE: list[tuple[str, int, str]] = [
    # (achievement_key, tier, prereq_hint)
    ("collect_wood",       1, "chop tree"),
    ("collect_sapling",    1, "pick up sapling"),
    ("collect_drink",      1, "drink from water"),
    ("eat_plant",          1, "eat a plant"),
    ("eat_cow",            1, "kill & eat cow"),
    ("defeat_zombie",      1, "attack zombie"),
    ("wake_up",            1, "sleep until full energy"),
    ("place_stone",        2, "have stone → do"),
    ("place_table",        2, "2 wood → place"),
    ("make_wood_pickaxe",  2, "table + wood"),
    ("make_wood_sword",    2, "table + wood"),
    ("collect_stone",      2, "wood pickaxe"),
    ("place_plant",        2, "sapling + dirt"),
    ("collect_coal",       3, "stone pickaxe"),
    ("make_stone_pickaxe", 3, "table + stone"),
    ("make_stone_sword",   3, "table + stone"),
    ("place_furnace",      3, "8 stone + table"),
    ("defeat_skeleton",    3, "attack skeleton"),
    ("collect_iron",       4, "stone pickaxe"),
    ("make_iron_pickaxe",  4, "furnace + iron"),
    ("make_iron_sword",    4, "furnace + iron"),
    ("collect_diamond",    4, "iron pickaxe"),
]

_TIER_META = {
    1: ("Basic",  "#4CAF50"),
    2: ("Early",  "#2196F3"),
    3: ("Mid",    "#FF9800"),
    4: ("Late",   "#F44336"),
}

ach_set = set(metrics.get("achievements_list", []))
n_unlocked = len(ach_set)
st.subheader(f"Achievement Map  —  {n_unlocked} / 22 unlocked")

for tier in (1, 2, 3, 4):
    tier_items = [(k, hint) for k, t, hint in _TECH_TREE if t == tier]
    label, color = _TIER_META[tier]
    done = sum(1 for k, _ in tier_items if k in ach_set)
    total = len(tier_items)

    st.markdown(
        f"<span style='color:{color};font-weight:700;font-size:0.95em'>"
        f"{label} Tier</span>"
        f"<span style='color:#888;font-size:0.85em'> — {done}/{total}</span>",
        unsafe_allow_html=True,
    )

    n_cols = min(total, 4)
    cols = st.columns(n_cols)
    for i, (key, hint) in enumerate(tier_items):
        name = key.replace("_", " ").title()
        col = cols[i % n_cols]
        if key in ach_set:
            col.markdown(
                f"<div style='background:#1b3a1f;border:1px solid {color};"
                f"border-radius:6px;padding:4px 8px;margin-bottom:4px;"
                f"font-size:0.82em;color:{color}'>✅ {name}</div>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f"<div style='background:#1a1a2e;border:1px solid #333;"
                f"border-radius:6px;padding:4px 8px;margin-bottom:4px;"
                f"font-size:0.82em;color:#666'>⬜ {name}"
                f"<br><span style='color:#444;font-size:0.8em'>{hint}</span></div>",
                unsafe_allow_html=True,
            )

# ── What's next ───────────────────────────────────────────────────────────────
remaining = [k for k, t, _ in _TECH_TREE if k not in ach_set]
if remaining:
    # Find the lowest tier not yet fully cleared
    next_tier = min(t for k, t, _ in _TECH_TREE if k not in ach_set)
    next_up = [k.replace("_", " ").title() for k, t, _ in _TECH_TREE
               if t == next_tier and k not in ach_set]
    _, next_color = _TIER_META[next_tier]
    st.markdown(
        f"<div style='margin-top:8px;padding:8px 12px;background:#111;"
        f"border-left:3px solid {next_color};border-radius:4px;"
        f"font-size:0.85em;color:#aaa'>"
        f"<b style='color:{next_color}'>Next up:</b> "
        f"{', '.join(next_up)}</div>",
        unsafe_allow_html=True,
    )
else:
    st.success("🏆 All 22 achievements unlocked!")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(REFRESH_S)
st.rerun()
