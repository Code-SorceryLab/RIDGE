"""RIDGE Post-Training Analysis Dashboard.

Standalone Streamlit app for analysing completed training runs.

Features:
  - Runs table summarising every condition (status, seeds, latest step, score)
  - Cross-condition spider/radar chart with 4 views:
      * Achievement categories (Collection/Survival/Crafting/Combat)
      * Training metrics (Crafter score, achievements, value stability, entropy)
      * Mean persona weights (Explorer/Survivor/Craftsman/Warrior)
      * All 22 achievements
  - Interactive legend toggle (Plotly) + RIDGE sharpness dropdown
  - One-click export of all 4 views to PNG + PDF via matplotlib

Launch with:
    streamlit run viewer/post_training_dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from viewer.spider_data import (
    ALL_ACHIEVEMENTS,  # noqa: F401  (re-exported for downstream tools)
    FIXED_BASELINE_KEYS,
    SHARPNESS_KEYS,
    SPIDER_CONDITIONS,
    VIEW_META,
    available_conditions,
    compute_view_data,
    find_run_dirs,
    load_condition as _load_condition_raw,
)


# ─── Plotly renderer (interactive) ────────────────────────────────────────────

def _hex_to_rgba(hex_str: str, alpha: float) -> str:
    h = hex_str.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def plotly_radar(title: str, axis_labels: list[str],
                  series: list[tuple[str, list[float], str]]):
    import plotly.graph_objects as go
    fig = go.Figure()
    theta = axis_labels + [axis_labels[0]]
    for label, values, colour in series:
        r = list(values) + [values[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta, name=label,
            line=dict(color=colour, width=2.5),
            marker=dict(size=6, color=colour),
            fill="toself",
            fillcolor=_hex_to_rgba(colour, 0.15),
            hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#444", linecolor="#666"),
            angularaxis=dict(gridcolor="#444", linecolor="#666"),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="v", x=1.06, y=0.5, yanchor="middle"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"),
        margin=dict(t=70, b=30, l=30, r=30),
        height=560,
    )
    return fig


@st.cache_data(ttl=60, show_spinner=False)
def load_condition(log_dir: str, prefix: str):
    return _load_condition_raw(log_dir, prefix)


# ─── Page layout ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RIDGE — Post-Training Analysis",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## Analysis Settings")
    log_dir     = st.text_input("TensorBoard log dir", "tensorboard_logs")
    results_dir = st.text_input("Export directory",    "results")
    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Caches expire automatically every 60s.")

st.title("📊 RIDGE — Post-Training Analysis")
st.caption("Cross-condition comparison of completed training runs. Use this after training finishes.")

# ── Runs table ────────────────────────────────────────────────────────────────
st.subheader("Runs")

table_rows = []
for cond, (label, _) in SPIDER_CONDITIONS.items():
    dirs = find_run_dirs(log_dir, cond)
    if not dirs:
        table_rows.append({
            "Condition": label, "Status": "⚠ missing", "Seeds": 0,
            "Latest Step": "—", "Crafter Score": "—", "Achievements": "—",
        })
        continue
    d = load_condition(log_dir, cond)
    if d is None:
        table_rows.append({
            "Condition": label, "Status": "⚠ error", "Seeds": len(dirs),
            "Latest Step": "—", "Crafter Score": "—", "Achievements": "—",
        })
        continue
    step    = d.get("max_step", 0)
    score   = d["metrics"].get("episode/crafter_score")
    ach_cum = d["metrics"].get("achievements/cumulative")
    status  = "✓ done" if step >= 950_000 else ("⏳ training" if step > 0 else "⚠ no data")
    table_rows.append({
        "Condition":     label,
        "Status":        status,
        "Seeds":         d["n_seeds"],
        "Latest Step":   f"{step:,}" if step else "—",
        "Crafter Score": f"{score:.4f}"        if score   is not None else "—",
        "Achievements":  f"{ach_cum:.1f}/22"  if ach_cum is not None else "—",
    })

st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

# ── Spider chart section ──────────────────────────────────────────────────────
st.divider()
st.subheader("🕸️ Cross-Condition Spider Chart")

present = available_conditions(log_dir)
if not present:
    st.info(f"No conditions found under `{log_dir}/`. Run training first, then refresh.")
    st.stop()

c1, c2 = st.columns([1, 2])
present_sharp = [k for k in SHARPNESS_KEYS if k in present]
if present_sharp:
    sharpness_choice = c1.selectbox(
        "RIDGE sharpness",
        present_sharp,
        format_func=lambda k: SPIDER_CONDITIONS[k][0],
        index=min(2, len(present_sharp) - 1),
        key="sharpness",
    )
else:
    sharpness_choice = None
    c1.caption("No RIDGE conditions completed yet.")

present_fixed = [c for c in FIXED_BASELINE_KEYS if c in present]
c2.caption(
    f"Click any condition in the legend to toggle it on/off.  "
    f"Showing {len(present_fixed)}/4 baselines"
    + (f" + {SPIDER_CONDITIONS[sharpness_choice][0]}" if sharpness_choice else "")
)

shown: list[str] = []
if sharpness_choice is not None:
    shown.append(sharpness_choice)
shown.extend(present_fixed)

data_by_cond: dict[str, dict] = {}
for c in shown:
    d = load_condition(log_dir, c)
    if d is not None:
        data_by_cond[c] = d

tabs = st.tabs([v[1] for v in VIEW_META])
for tab, (view_key, _, view_subtitle) in zip(tabs, VIEW_META):
    with tab:
        axis_labels, series = compute_view_data(view_key, shown, data_by_cond)
        if not series:
            st.caption("No data for this view yet.")
            continue
        fig = plotly_radar(view_subtitle, axis_labels, series)
        if view_key == "all_22":
            fig.update_layout(height=640, polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#444"),
                angularaxis=dict(gridcolor="#444", tickfont=dict(size=10)),
                bgcolor="rgba(0,0,0,0)",
            ))
        st.plotly_chart(fig, use_container_width=True, key=f"radar_{view_key}")

# ── Export section ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Export")
exp_c1, exp_c2 = st.columns([1, 3])
if exp_c1.button("📥 Save spider figure (PNG + PDF)", use_container_width=True, type="primary"):
    from viewer.dashboard import generate_spider_plots
    try:
        paths = generate_spider_plots(
            log_dir=log_dir,
            sharpness_condition=sharpness_choice,
            out_dir=results_dir,
        )
        st.success(f"✓ Saved spider figure to `{results_dir}/spider_combined.{{png,pdf}}`")
        with st.expander("Files written"):
            for p in paths:
                st.code(p)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Export failed: {exc}")
exp_c2.caption(
    "One consolidated 2x2 figure showing all 4 views — saved as `spider_combined.png` and "
    "`spider_combined.pdf` via matplotlib polar projection. Shared legend at the bottom."
)
