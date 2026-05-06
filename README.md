# RIDGE — Reactive Inter-persona Dynamic Goal Engine

State-conditioned reward blending for behavioral coverage in deep RL game agents.

A single PPO agent blends Explorer, Survivor, and Craftsman persona reward weights via smooth sigmoid functions conditioned on internal game state. Trained on the [Crafter](https://github.com/danijar/crafter) environment (22 achievements).

**Authors:** Al Shifan, Kevin Chua, Cristiano Politowski — Code Sorcery Lab

---

## Research Questions

| | Question |
|---|---|
| RQ1 | Does state-conditioned blending match multi-persona ensemble coverage on Crafter's 22 achievements? |
| RQ2 | Does it do so at lower training compute (steps-to-coverage)? |
| RQ3 | Does smooth sigmoid blending avoid the switching-stability dilemma that hard switching causes? |

**Experimental sweep:** 4 conditions × 5 seeds × 1M Crafter steps.

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, Crafter, TensorBoard, pygame, tqdm, matplotlib.

> On Windows, if `crafter` fails to build:
> ```powershell
> $env:PYTHONUTF8="1"; pip install -r requirements.txt
> ```

---

## Quick Start

```bash
python menu.py
```

```
╔══════════════════════════════════════╗
║          RIDGE — Main Menu           ║
╠══════════════════════════════════════╣
║  1. Train RIDGE (adaptive blending)  ║
║  2. Train Explorer baseline          ║
║  3. Train Survivor baseline          ║
║  4. Train Craftsman baseline         ║
║  5. Train all conditions (sweep)     ║
║  6. Live Viewer (watch agent play)   ║
║  7. Launch TensorBoard               ║
║  8. Run comparison graphs            ║
║  9. Evaluate checkpoint              ║
║  0. Exit                             ║
╚══════════════════════════════════════╝
```

---

## Direct CLI Usage

```bash
# Train with a specific config
python scripts/train.py --config configs/ridge_blend.yaml --seed 42

# Train with live viewer
python scripts/train.py --config configs/ridge_blend.yaml --live

# Evaluate a checkpoint
python scripts/evaluate.py --config configs/ridge_blend.yaml --checkpoint checkpoints/ridge_adaptive_seed42/best.pt --episodes 20

# Generate comparison plots
python scripts/compare.py --logdir tensorboard_logs --out results
```

---

## Project Structure

```
ridge-rlvg2026/
├── menu.py                 # Entry point — run this
├── requirements.txt
├── configs/
│   ├── default.yaml        # Base hyperparameters
│   ├── ridge_blend.yaml    # RIDGE adaptive blending
│   ├── explorer.yaml       # Fixed Explorer baseline
│   ├── survivor.yaml       # Fixed Survivor baseline
│   └── craftsman.yaml      # Fixed Craftsman baseline
├── ridge/
│   ├── game.py             # Crafter wrapper, state extraction
│   ├── agent.py            # PPO agent, multi-head critic
│   ├── rewards.py          # Persona rewards, sigmoid blending engine
│   ├── trainer.py          # Training loop, TensorBoard logging
│   ├── menu.py             # Menu implementation
│   └── utils.py            # Config loading, seeding, device
├── viewer/
│   ├── live_viewer.py      # Real-time pygame viewer with debug overlay
│   └── dashboard.py        # TensorBoard launcher, comparison plots
├── scripts/
│   ├── train.py            # CLI training entry point
│   ├── evaluate.py         # Checkpoint evaluation
│   └── compare.py          # Multi-run comparison
└── tests/
    ├── test_rewards.py
    ├── test_agent.py
    └── test_game.py
```

---

## Architecture

### Multi-Head Critic
RIDGE uses a shared CNN encoder with **three separate value heads** — one per persona. Each head estimates value under its own stationary reward signal. The aggregate value fed to PPO is:

```
V = w_e * V_explorer + w_s * V_survivor + w_c * V_craftsman
```

This keeps each value head's target distribution stationary even as the blended reward shifts, directly addressing PPO non-stationarity under dynamic reward weighting.

### Sigmoid Blending
Persona weights are computed via smooth sigmoid functions over the game state vector `[health, food, drink, energy, progress, tool_progress]`. No hard switches — weights transition smoothly as conditions change.

---

## Running Tests

```bash
pytest tests/test_rewards.py tests/test_agent.py   # no Crafter needed
pytest tests/                                       # full suite (requires Crafter)
```

---

## Configs

All hyperparameters live in `configs/default.yaml`. Condition-specific configs override only what they need. No hardcoded values in code.

Key defaults: `lr=3e-4`, `gamma=0.99`, `rollout_steps=256`, `ppo_epochs=4`, `total_steps=1_000_000`.
