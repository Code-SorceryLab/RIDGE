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

# ── Achievements unlocked ─────────────────────────────────────────────────────
ach_list = metrics.get("achievements_list", [])
st.subheader(f"Achievements Unlocked ({len(ach_list)} / 22)")

if ach_list:
    cols = st.columns(4)
    for i, ach in enumerate(sorted(ach_list)):
        cols[i % 4].success(ach.replace("_", " ").title())
else:
    st.caption("None yet — keep training!")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(REFRESH_S)
st.rerun()
