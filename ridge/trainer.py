"""Training loop: rollout collection, PPO updates, TensorBoard logging, checkpointing."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console as RichConsole
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.tensorboard import SummaryWriter

from ridge.agent import PPOAgent, RolloutBuffer
from ridge.game import CrafterWrapper, make_env
from ridge.rewards import compute_blended_reward
from ridge.utils import ensure_dir, get_device, set_seeds

logger = logging.getLogger(__name__)
_console = RichConsole()

ALL_ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]

class Trainer:
    """Manages the full RIDGE training loop.

    Handles rollout collection, advantage estimation, PPO updates,
    TensorBoard logging, and checkpoint management.
    """

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        """Initialise Trainer from config.

        Args:
            config: Project config dict loaded from YAML.
            seed: Optional seed override; falls back to config['seed'].
        """
        self._config = config
        self._seed = seed if seed is not None else int(config.get("seed", 42))

        set_seeds(self._seed)
        self._device = get_device()

        self._env: CrafterWrapper = make_env(config, seed=self._seed)
        num_actions: int = self._env.action_space.n

        self._agent = PPOAgent(config, num_actions=num_actions, device=self._device)
        self._buffer = RolloutBuffer()

        # Logging / checkpointing paths
        run_name: str = config.get("run_name", "ridge")
        log_dir: str = config.get("log_dir", "tensorboard_logs")
        ckpt_dir: str = config.get("checkpoint_dir", "checkpoints")

        self._run_dir = str(Path(log_dir) / f"{run_name}_seed{self._seed}")
        self._ckpt_dir = str(Path(ckpt_dir) / f"{run_name}_seed{self._seed}")
        ensure_dir(self._run_dir)
        ensure_dir(self._ckpt_dir)

        self._writer = SummaryWriter(log_dir=self._run_dir)
        self._total_steps: int = int(config.get("total_steps", 1_000_000))
        self._rollout_steps: int = int(config.get("rollout_steps", 256))
        self._checkpoint_every: int = int(config.get("checkpoint_every", 50_000))
        self._live_view: bool      = bool(config.get("live_view",      False))
        self._live_dashboard: bool = bool(config.get("live_dashboard", False))
        self._auto_resume: bool    = bool(config.get("auto_resume",    True))
        self._render_every: int    = max(1, int(config.get("render_every", 4)))

        if self._live_view:
            try:
                from viewer.live_viewer import LiveViewer
                if LiveViewer._instance is None:
                    LiveViewer(render_fps=int(config.get("render_fps", 15)))
                    _console.print(
                        "  [bold green]Live viewer open[/] — ↑↓ speed  |  close window to stop"
                    )
            except Exception as exc:
                logger.warning("Could not open live viewer: %s", exc)
                self._live_view = False

        self._global_step: int = 0
        self._episode_count: int = 0
        self._best_achievement_count: int = 0
        self._episode_persona_totals: dict[str, float] = {
            "explorer": 0.0, "survivor": 0.0, "craftsman": 0.0, "warrior": 0.0
        }
        self._cumulative_persona_totals: dict[str, float] = {
            "explorer": 0.0, "survivor": 0.0, "craftsman": 0.0, "warrior": 0.0
        }
        self._kl_high_count: int     = 0
        self._entropy_low_count: int = 0

        # PyTorch CPU thread configuration
        if "num_threads" in config:
            n = int(config["num_threads"])
            torch.set_num_threads(n)
            try:
                torch.set_num_interop_threads(max(1, n // 2))
            except RuntimeError:
                pass  # can only be set once per process; subsequent Trainers skip it

        self._cumulative_achievements: set[str] = set()

        # Rolling window for per-achievement success rate (last 100 episodes)
        self._achievement_window: list[list[str]] = []

        logger.info(
            "Trainer ready — mode=%s seed=%d device=%s",
            config.get("blending_mode", "ridge"),
            self._seed,
            self._device,
        )

        if self._auto_resume:
            self._try_auto_resume()

        # Mute game logger during training to prevent it from breaking the progress bar
        logging.getLogger("ridge.game").setLevel(logging.WARNING)

    # -------------------------------------------------------------------------

    def _find_latest_checkpoint(self) -> str | None:
        """Scan checkpoint dir for the highest-step step_N.pt file."""
        ckpt_dir = Path(self._ckpt_dir)
        if not ckpt_dir.exists():
            return None
        candidates = []
        for p in ckpt_dir.glob("step_*.pt"):
            try:
                step = int(p.stem.split("_")[1])
                candidates.append((step, str(p)))
            except (IndexError, ValueError):
                pass
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    def _try_auto_resume(self) -> None:
        """Load the latest periodic checkpoint if one exists."""
        ckpt = self._find_latest_checkpoint()
        if ckpt is None:
            return
        step = int(Path(ckpt).stem.split("_")[1])
        self._agent.load_checkpoint(ckpt)
        self._global_step = step
        _console.print(
            f"  [bold yellow]↩ Auto-resumed from[/] {ckpt}  "
            f"[dim](step {step:,})[/dim]"
        )

    def _write_live_metrics(
        self, info: dict[str, Any], weights: np.ndarray, fps: float
    ) -> None:
        """Write JSON metrics + game frame for the Streamlit dashboard."""
        import json as _json
        out_dir = Path("training_live")
        out_dir.mkdir(exist_ok=True)
        payload = {
            "step":             self._global_step,
            "episode":          self._episode_count,
            "fps":              round(fps, 1),
            "score":            round(self._compute_crafter_score(), 5),
            "achievements":     len(self._cumulative_achievements),
            "weights":          weights.tolist(),
            "vitals": {
                "health": info.get("health", 9),
                "food":   info.get("food",   9),
                "drink":  info.get("drink",  9),
                "energy": info.get("energy", 9),
            },
            "persona_rewards":  dict(self._episode_persona_totals),
            "achievements_list": sorted(self._cumulative_achievements),
        }
        tmp = out_dir / "metrics.json.tmp"
        try:
            with open(tmp, "w") as f:
                _json.dump(payload, f)
            tmp.replace(out_dir / "metrics.json")
        except OSError:
            # Windows: Streamlit may hold metrics.json open; skip this update.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return

        frame = self._env.render()
        if frame is not None:
            try:
                np.save(str(out_dir / "frame.npy"), frame)
            except OSError:
                pass

    def _check_training_health(self, metrics: dict[str, float]) -> None:
        """Warn if KL divergence is spiking or entropy has collapsed."""
        kl = metrics.get("kl_divergence", 0.0)
        ent = metrics.get("entropy", 1.0)

        if kl > 0.05:
            self._kl_high_count += 1
            if self._kl_high_count >= 3:
                _console.print(
                    f"  [bold red]⚠  KL={kl:.3f} for {self._kl_high_count} updates "
                    f"— policy may be destabilising[/]"
                )
        else:
            self._kl_high_count = 0

        if ent < 0.01:
            self._entropy_low_count += 1
            if self._entropy_low_count >= 3:
                _console.print(
                    f"  [bold yellow]⚠  Entropy={ent:.4f} — agent may be stuck "
                    f"in a local optimum[/]"
                )
        else:
            self._entropy_low_count = 0

    def _print_training_summary(self, wall_seconds: float) -> None:
        """Print a rich end-of-training summary panel."""
        from rich.panel import Panel
        from rich.table import Table
        from rich import box as rbox

        h = int(wall_seconds // 3600)
        m = int((wall_seconds % 3600) // 60)
        s = int(wall_seconds % 60)
        elapsed_str = f"{h}h {m}m {s}s"

        t = Table(box=rbox.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("key",   style="dim white",   width=24)
        t.add_column("value", style="bold white")

        t.add_row("Steps completed",    f"{self._global_step:,} / {self._total_steps:,}")
        t.add_row("Episodes",           f"{self._episode_count:,}")
        t.add_row("Wall time",          elapsed_str)
        t.add_row("Crafter score",      f"{self._compute_crafter_score():.4f}")
        t.add_row("Achievements seen",  f"{len(self._cumulative_achievements)} / 22")
        t.add_row("Best episode (ach)", f"{self._best_achievement_count}")
        t.add_row("Checkpoints saved",  self._ckpt_dir)
        t.add_section()
        for persona in ("explorer", "survivor", "craftsman", "warrior"):
            r = self._cumulative_persona_totals.get(persona, 0.0)
            t.add_row(f"  {persona.capitalize()} reward (total)", f"{r:,.1f}")

        _console.print(Panel(t, title="[bold cyan]Training Complete[/]", border_style="cyan"))

    # -------------------------------------------------------------------------

    def _fresh_unlocked(self) -> dict[str, set]:
        """Empty per-persona achievement tracker for a new episode.

        Passed into compute_blended_reward so each achievement bonus
        fires at most once per episode per persona.
        """
        return {"explorer": set(), "survivor": set(), "craftsman": set(), "warrior": set()}

    # -------------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop until total_steps is reached."""
        train_start = time.time()

        obs, info = self._env.reset()
        current_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        episode_start_step = 0
        episode_unlocked = self._fresh_unlocked()

        fps_timer = time.time()
        fps_step_count = 0
        fps = 0.0

        progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold cyan]RIDGE[/bold cyan]"),
            BarColumn(
                bar_width=None,
                style="dark_green",
                complete_style="green",
                finished_style="bright_green",
            ),
            TaskProgressColumn(style="bold white"),
            MofNCompleteColumn(),
            TextColumn("[dim]·[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]<[/dim]"),
            TimeRemainingColumn(),
            TextColumn("[dim]·[/dim]"),
            TextColumn("[yellow]{task.fields[fps]}fps[/yellow]"),
            TextColumn("[dim]|[/dim]"),
            TextColumn("ep:[cyan]{task.fields[ep]}[/cyan]"),
            TextColumn("score:[magenta]{task.fields[score]}[/magenta]"),
            TextColumn("A:[green]{task.fields[achiev]}[/green]"),
            console=_console,
            transient=False,
        )
        task_id = progress.add_task(
            "RIDGE",
            total=self._total_steps,
            completed=self._global_step,   # non-zero when auto-resumed
            fps="0",
            ep="0",
            score="0.000",
            achiev="0/22",
        )
        progress.start()

        try:
            while self._global_step < self._total_steps:
                # Collect rollout
                # ------------------------------------------------------------------
                self._buffer.clear()
                update_start = time.time()

                for _ in range(self._rollout_steps):
                    state_vec = self._env.extract_state_vector(info)

                    blended_reward, weights, per_persona = compute_blended_reward(
                        info, state_vec, self._config, episode_unlocked
                    )
                    current_weights = weights
                    for k, v in per_persona.items():
                        self._episode_persona_totals[k]     += v
                        self._cumulative_persona_totals[k]  += v

                    action, log_prob, value, per_head_val = self._agent.select_action(obs, weights)

                    next_obs, _, terminated, truncated, next_info = self._env.step(action)
                    done = terminated or truncated

                    self._env.update_episode_stats(blended_reward, next_info, weights)

                    per_persona_arr = np.array([
                        per_persona["explorer"],
                        per_persona["survivor"],
                        per_persona["craftsman"],
                        per_persona["warrior"],
                    ], dtype=np.float32)

                    self._buffer.obs.append(obs.copy())
                    self._buffer.actions.append(action)
                    self._buffer.log_probs.append(log_prob)
                    self._buffer.values.append(value)
                    self._buffer.per_head_values.append(per_head_val)          # V_i — for bootstrap
                    self._buffer.persona_step_rewards.append(per_persona_arr)  # r_i — for G_i targets
                    self._buffer.rewards.append(blended_reward)
                    self._buffer.dones.append(done)
                    self._buffer.persona_weights.append(weights.copy())
                    self._buffer.infos.append(next_info)

                    if self._live_view:
                        self._push_live_frame(obs, next_info, weights)

                    self._global_step += 1
                    fps_step_count += 1

                    if done:
                        stats = self._env.get_episode_stats().to_dict()
                        self._log_episode(stats, self._episode_persona_totals, weights)
                        self._episode_count += 1
                        self._episode_persona_totals = {
                            "explorer": 0.0, "survivor": 0.0,
                            "craftsman": 0.0, "warrior": 0.0,
                        }

                        obs, info = self._env.reset()
                        current_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
                        episode_start_step = self._global_step
                        episode_unlocked = self._fresh_unlocked()
                    else:
                        obs = next_obs
                        info = next_info

                    if self._global_step >= self._total_steps:
                        break

                # ------------------------------------------------------------------
                # PPO update
                # ------------------------------------------------------------------
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        obs, dtype=torch.float32, device=self._device
                    ).unsqueeze(0)
                    w_t = torch.as_tensor(
                        current_weights, dtype=torch.float32, device=self._device
                    )
                    _, last_value_t, last_per_head_t = self._agent.network(obs_t, w_t)
                    last_value        = float(last_value_t.squeeze().item())
                    last_per_head_val = last_per_head_t.squeeze(0).cpu().numpy()  # (3,)

                advantages, returns, per_persona_returns = self._agent.compute_advantages(
                    self._buffer, last_value, last_per_head_val
                )
                update_metrics = self._agent.ppo_update(
                    self._buffer, advantages, returns, per_persona_returns
                )

                update_time = time.time() - update_start

                now = time.time()
                elapsed = now - fps_timer
                fps = fps_step_count / elapsed if elapsed > 0 else 0.0
                fps_timer = now
                fps_step_count = 0

                self._log_update(update_metrics, fps, update_time)
                self._check_training_health(update_metrics)

                if self._global_step % self._checkpoint_every < self._rollout_steps:
                    self._save_periodic_checkpoint()

                if self._live_dashboard:
                    try:
                        self._write_live_metrics(info, current_weights, fps)
                    except Exception:
                        pass

                steps_advanced = min(
                    self._rollout_steps,
                    self._total_steps - (self._global_step - self._rollout_steps),
                )
                progress.update(
                    task_id,
                    advance=steps_advanced,
                    fps=f"{fps:.0f}",
                    ep=str(self._episode_count),
                    score=f"{self._compute_crafter_score():.3f}",
                    achiev=f"{len(self._cumulative_achievements)}/22",
                )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        finally:
            progress.stop()
            self._writer.close()
            self._env.close()
            logger.info("Training complete — %d steps, %d episodes", self._global_step, self._episode_count)
            self._print_training_summary(time.time() - train_start)

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _compute_crafter_score(self) -> float:
        """Official Crafter score (Hafner 2022): geometric mean with +1% offset."""
        if not self._achievement_window:
            return 0.0
        n = len(self._achievement_window)
        # percentages in [0, 100]
        rates_pct = np.array([
            100 * sum(1 for ep in self._achievement_window if ach in ep) / n
            for ach in ALL_ACHIEVEMENTS
        ])
        return float(np.exp(np.mean(np.log(1 + rates_pct))) - 1)  # returns a percentage

    def _log_episode(
        self,
        stats: dict[str, Any],
        per_persona: dict[str, float],
        weights: np.ndarray,
    ) -> None:
        """Write per-episode metrics to TensorBoard.

        Args:
            stats: Episode stats dict from EpisodeStats.to_dict().
            per_persona: Per-persona scalar rewards for this step.
            weights: Persona weights array of shape (3,).
        """
        s = self._global_step
        w = self._writer

        w.add_scalar("reward/episode_total",    stats["total_reward"],    s)
        w.add_scalar("reward/ep_explorer",  per_persona["explorer"],  s)
        w.add_scalar("reward/ep_survivor",  per_persona["survivor"],  s)
        w.add_scalar("reward/ep_craftsman", per_persona["craftsman"], s)
        w.add_scalar("reward/ep_warrior",   per_persona["warrior"],   s)

        mean_w = stats["mean_weights"]
        w.add_scalar("weights/explorer",  mean_w[0], s)
        w.add_scalar("weights/survivor",  mean_w[1], s)
        w.add_scalar("weights/craftsman", mean_w[2], s)
        w.add_scalar("weights/warrior",   mean_w[3], s)

        ach_count = stats["achievement_count"]
        w.add_scalar("achievements/count", ach_count, s)

        for name in stats["achievements_unlocked"]:
            self._cumulative_achievements.add(name)
        w.add_scalar("achievements/cumulative", len(self._cumulative_achievements), s)

        self._achievement_window.append(stats["achievements_unlocked"])
        if len(self._achievement_window) > 100:
            self._achievement_window.pop(0)
        for ach_name in self._cumulative_achievements:
            rate = sum(1 for ep in self._achievement_window if ach_name in ep) / len(self._achievement_window)
            w.add_scalar(f"achievements/{ach_name}", rate, s)

        w.add_scalar("episode/length", stats["steps"], s)

        crafter_score = self._compute_crafter_score()
        w.add_scalar("episode/crafter_score", crafter_score, s)

        if ach_count > self._best_achievement_count:
            self._best_achievement_count = ach_count
            self._agent.save_checkpoint(str(Path(self._ckpt_dir) / "best.pt"))
            logger.info("New best: %d achievements at step %d", ach_count, s)

    def _log_update(self, metrics: dict[str, float], fps: float, update_time: float) -> None:
        """Write PPO update metrics to TensorBoard.

        Args:
            metrics: Dict of loss/diagnostic metrics from ppo_update.
            fps: Frames per second over the last rollout.
            update_time: Wall-clock seconds for the update.
        """
        s = self._global_step
        w = self._writer
        w.add_scalar("agent/policy_loss",          metrics["policy_loss"],          s)
        w.add_scalar("agent/value_loss",           metrics["value_loss"],           s)
        w.add_scalar("agent/entropy",              metrics["entropy"],              s)
        w.add_scalar("agent/value_loss_explorer",  metrics["value_loss_explorer"],  s)
        w.add_scalar("agent/value_loss_survivor",  metrics["value_loss_survivor"],  s)
        w.add_scalar("agent/value_loss_craftsman", metrics["value_loss_craftsman"], s)
        w.add_scalar("agent/value_loss_warrior",   metrics["value_loss_warrior"],   s)
        w.add_scalar("agent/kl_divergence",        metrics["kl_divergence"],        s)
        w.add_scalar("agent/clip_fraction",        metrics["clip_fraction"],        s)
        w.add_scalar("perf/fps",                   fps,                             s)
        w.add_scalar("perf/update_time",           update_time,                     s)

    def _save_periodic_checkpoint(self) -> None:
        path = str(Path(self._ckpt_dir) / f"step_{self._global_step}.pt")
        self._agent.save_checkpoint(path)

    def _push_live_frame(
        self, obs: np.ndarray, info: dict[str, Any], weights: np.ndarray
    ) -> None:
        """Send frame and debug info to the live viewer if active."""
        if self._global_step % self._render_every != 0:
            return
        try:
            from viewer.live_viewer import LiveViewer
            still_open = LiveViewer.push_frame(self._env.render(), info, weights)
            if not still_open:
                self._live_view = False
        except Exception:
            pass