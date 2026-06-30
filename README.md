# RIDGE: Reactive Inter-persona Dynamic Goal Engine

State-conditioned reward blending for behavioral coverage in deep RL game agents.

A single PPO agent blends Explorer, Survivor, Craftsman, and Warrior persona
reward weights using smooth sigmoid functions conditioned on internal game
state. Trained on the [Crafter](https://github.com/danijar/crafter) environment
(22 achievements).

**Authors:** Al Shifan, Kevin Chua, Cristiano Politowski (Code Sorcery Lab)

Accompanies the IEEE CoG 2026 short paper *RIDGE: State-Conditioned Reward
Blending for Behavioral Coverage in Deep RL Game Agents*. Code, data, and
trained checkpoints are archived on Zenodo (DOI: 10.5281/zenodo.20192914).

---

## Research Questions

| | Question |
|---|---|
| RQ1 | Does state-conditioned blending match multi-persona ensemble coverage on Crafter's 22 achievements? |
| RQ2 | Does it do so at lower training compute (steps to coverage)? |
| RQ3 | Does smooth sigmoid blending avoid the switching-stability dilemma that hard switching causes? |

**Conditions:** RIDGE (adaptive, sharpness 1.0), four fixed-persona baselines
(Explorer, Survivor, Craftsman, Warrior), an all-ones constant-reward sanity
floor, and a sharpness ablation (sharpness 0.0, 0.5, 1.0, 1.5, 2.0). Budget:
1M Crafter steps per run.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10 or newer. Core dependencies: PyTorch, Crafter,
TensorBoard, pygame, rich, NumPy, and Matplotlib. The full list is in
`requirements.txt`.

> On Windows, if `crafter` fails to build:
> ```powershell
> $env:PYTHONUTF8="1"; pip install -r requirements.txt
> ```

---

## Quick Start

```bash
python menu.py
```

This opens an interactive menu. Main options:

- **1 to 5:** train RIDGE or a fixed-persona baseline (Explorer, Survivor, Craftsman, Warrior)
- **A:** train the all-ones constant-reward sanity floor
- **6:** run all baseline conditions back to back
- **11:** run the blend sharpness ablation (0.0, 0.5, 1.0, 1.5, 2.0)
- **M:** multi-seed run for cross-seed confidence intervals
- **7 to 10:** live viewer, TensorBoard, comparison plots, checkpoint evaluation
- **L, P:** Streamlit live and post-training dashboards

---

## Command Line Usage

```bash
# Train one condition (skips the menu)
python scripts/train.py --config configs/ridge_blend.yaml --seed 42

# Train with the live viewer
python scripts/train.py --config configs/ridge_blend.yaml --live

# Evaluate a checkpoint
python scripts/evaluate.py --config configs/ridge_blend.yaml \
    --checkpoint checkpoints/ridge_adaptive_seed42/best.pt --episodes 20

# Generate comparison plots from logs
python scripts/compare.py --logdir tensorboard_logs --out results

# Sharpness ablation (RQ3)
python scripts/sharpness_sweep.py --config configs/ridge_blend.yaml \
    --sharpness 0.0 0.5 1.0 1.5 2.0 --seed 42

# Multi-seed run for confidence intervals
python scripts/run_multi_seed.py \
    --configs configs/ridge_sharp100.yaml configs/ridge_sharp150.yaml --seeds 1 2 3
```

---

## Project Structure

```
RIDGE/
  menu.py                 Entry point, run this
  requirements.txt
  configs/                YAML configs, one per condition
  ridge/
    game.py               Crafter wrapper, state extraction
    agent.py              PPO agent, multi-head critic
    rewards.py            Persona rewards, sigmoid blending engine
    trainer.py            Training loop, TensorBoard logging
    menu.py               Interactive menu implementation
    utils.py              Config loading, seeding, device
  viewer/                 Live pygame viewer, Streamlit dashboards, plots
  scripts/                Training, evaluation, comparison, ablation tools
  tests/                  Unit tests for rewards, agent, game
```

---

## Architecture

### Personas

Four designer-authored persona rewards, each scoring a transition from the
perspective of its play style:

- **Explorer:** novel-tile discovery and broad resource collection
- **Survivor:** continuous shaping over the vitals (health, food, drink, energy) plus survival milestones such as `wake_up`
- **Craftsman:** the crafting tech tree, scaled by depth (wood to iron to diamond)
- **Warrior:** weapon crafting and combat

A designer can read each reward and recognize the intended play style.

### State-Conditioned Blender

Persona weights are computed from smooth sigmoid functions over a
6-dimensional state vector:

```
[health, food, drink, energy, progress, tool_progress]   (each in [0, 1])
```

Raw sigmoid activations are normalized to a simplex, so the four weights sum to
1. There are no hard switches: weights shift smoothly as state changes. The
`blend_sharpness` parameter controls how abrupt the transitions are. At
sharpness 0 the weights collapse to a uniform mixture (0.25 each); larger
values approach hard switching. This is the primary RQ3 ablation knob.

### Multi-Head Critic

A shared CNN encoder feeds one policy head and four value heads, one per
persona. The aggregate value used by PPO is the weighted sum of the per-head
values:

```
V = w_e * V_explorer + w_s * V_survivor + w_c * V_craftsman + w_w * V_warrior
```

Each value head trains against its own per-persona return, so every head sees a
stationary target even as the blended reward shifts. This isolates
value-function learning from non-stationarity in the aggregate reward.

---

## Configs

All hyperparameters live in `configs/default.yaml`. Condition configs override
only what they change.

| Config | Condition |
|--------|-----------|
| `default.yaml` | Base hyperparameters, inherited by all |
| `ridge_blend.yaml` | RIDGE adaptive blending (sharpness 1.0) |
| `explorer.yaml`, `survivor.yaml`, `craftsman.yaml`, `warrior.yaml` | Fixed single-persona baselines |
| `all_ones.yaml` | Constant +1.0 reward, sanity floor |
| `ridge_sharp000.yaml` ... `ridge_sharp200.yaml` | Sharpness ablation (0.0 to 2.0) |

Key defaults: `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_epsilon=0.2`,
`entropy_coef=0.01`, `value_coef=0.5`, `ppo_epochs=4`, `num_minibatches=8`,
`rollout_steps=512`, `total_steps=1_000_000`.

---

## Running Tests

```bash
pytest tests/test_rewards.py tests/test_agent.py   # no Crafter needed
pytest tests/                                       # full suite (requires Crafter)
```
