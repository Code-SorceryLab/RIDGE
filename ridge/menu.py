"""Interactive CLI menu — main entry point for RIDGE."""

import logging
import os
import random
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ridge.utils import load_default_config, setup_logging

logger = logging.getLogger(__name__)
console = Console()

_LOGO = """\
 ██████╗  ██╗██████╗   ██████╗ ███████╗
 ██╔══██╗ ██║██╔══██╗ ██╔════╝ ██╔════╝
 ██████╔╝ ██║██║  ██║ ██║  ███╗█████╗
 ██╔══██╗ ██║██║  ██║ ██║   ██║██╔══╝
 ██║  ██║ ██║██████╔╝ ╚██████╔╝███████╗
 ╚═╝  ╚═╝ ╚═╝╚═════╝   ╚═════╝ ╚══════╝"""

_SUBTITLE = "Reinforcement Intelligence with Dynamic Goal Evolution"

_MENU_ITEMS = [
    ("1",  "Train RIDGE",          "Adaptive multi-persona blending"),
    ("2",  "Train Explorer",       "Exploration-focused baseline"),
    ("3",  "Train Survivor",       "Survival-focused baseline"),
    ("4",  "Train Craftsman",      "Crafting-focused baseline"),
    ("5",  "Train Warrior",        "Combat-focused baseline"),
    ("6",  "Sweep All Conditions", "Run all 5 conditions sequentially"),
    ("7",  "Live Viewer",          "Watch agent play in real time"),
    ("8",  "TensorBoard",          "Launch training dashboard"),
    ("9",  "Comparison Graphs",    "Generate plots from logs"),
    ("10", "Evaluate Checkpoint",  "Score a saved checkpoint"),
    ("11", "Sharpness Sweep",      "RQ3 ablation — blend_sharpness 0.5/1.0/2.0/4.0"),
    ("L",  "Live Dashboard",       "Open Streamlit training monitor in browser"),
    ("D",  "Delete All Models",    "Wipe checkpoints & logs — requires DELETE"),
    ("0",  "Exit",                 "Quit RIDGE"),
]

_CONFIG_MAP = {
    "1": "configs/ridge_blend.yaml",
    "2": "configs/explorer.yaml",
    "3": "configs/survivor.yaml",
    "4": "configs/craftsman.yaml",
    "5": "configs/warrior.yaml",
}

_SHARPNESS_MAP = {
    "0.5": "configs/ridge_sharp050.yaml",
    "1.0": "configs/ridge_sharp100.yaml",
    "2.0": "configs/ridge_sharp200.yaml",
    "4.0": "configs/ridge_sharp400.yaml",
}


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _print_header() -> None:
    logo = Text(_LOGO, style="bold cyan", justify="center")
    subtitle = Text(_SUBTITLE, style="italic dim white", justify="center")
    content = Text.assemble(logo, "\n", subtitle)
    console.print(Panel(content, border_style="cyan", padding=(0, 2)))


def _print_menu() -> None:
    _clear()
    _print_header()

    table = Table(
        box=box.ROUNDED,
        border_style="bright_cyan",
        show_header=False,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("key",         style="bold yellow", width=5, justify="center")
    table.add_column("option",      style="bold white",  min_width=22)
    table.add_column("description", style="dim white")

    for key, name, desc in _MENU_ITEMS[:-1]:
        table.add_row(f"[{key}]", name, desc)

    table.add_section()
    exit_key, exit_name, exit_desc = _MENU_ITEMS[-1]
    table.add_row(f"[{exit_key}]", exit_name, exit_desc, style="dim red")

    console.print(table)
    console.print()


def _prompt_config(default_path: str) -> dict[str, Any]:
    user_path = input(f"  Config [{default_path}]: ").strip()
    path = user_path if user_path else default_path
    return load_default_config(path)


def _prompt_seed() -> int:
    default_seed = random.randint(0, 99999)
    raw = input(f"  Seed [{default_seed}]: ").strip()
    try:
        return int(raw) if raw else default_seed
    except ValueError:
        console.print(f"  [yellow]Invalid seed — using {default_seed}[/yellow]")
        return default_seed


def _print_cpu_info() -> None:
    """Print CPU/thread info so the user knows what compute is available."""
    try:
        import psutil
        import torch
        phys = psutil.cpu_count(logical=False)
        logi = psutil.cpu_count(logical=True)
        torch_threads = torch.get_num_threads()
        console.print(
            f"  [dim]CPU: {phys} physical / {logi} logical cores  |  "
            f"PyTorch threads: {torch_threads}[/dim]"
        )
    except Exception:
        pass


def _start_dashboard_bg() -> None:
    """Launch Streamlit dashboard in the background — non-blocking, no new window."""
    import subprocess
    import sys

    dashboard = Path("viewer") / "streamlit_dashboard.py"
    if not dashboard.exists():
        console.print("  [red]streamlit_dashboard.py not found.[/red]")
        return
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(dashboard),
         "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    console.print(
        f"  [bold green]Live Dashboard →[/] http://localhost:8501  "
        f"[dim](PID {proc.pid})[/dim]"
    )


def _do_live_dashboard() -> None:
    """Menu option L — launch dashboard then wait for Enter."""
    _start_dashboard_bg()
    console.print("  [dim]Set live_dashboard: true in your config before training.[/dim]")
    input("  Press Enter to return to menu...")


def _run_training(config_path: str) -> None:
    from ridge.trainer import Trainer

    config = _prompt_config(config_path)
    seed = _prompt_seed()
    _print_cpu_info()
    if config.get("live_dashboard", False):
        _start_dashboard_bg()
    console.print(f"\n  [bold green]Starting:[/] {config.get('run_name', 'ridge')} | seed={seed}\n")
    trainer = Trainer(config, seed=seed)
    trainer.train()
    _post_training_prompt(config)


def _post_training_prompt(config: dict[str, Any]) -> None:
    choice = input("\n  Launch TensorBoard? [y/N]: ").strip().lower()
    if choice == "y":
        _do_launch_tensorboard(config.get("log_dir", "tensorboard_logs"))

    choice = input("  Generate comparison graphs? [y/N]: ").strip().lower()
    if choice == "y":
        _do_comparison_graphs(config.get("log_dir", "tensorboard_logs"))


def _run_sweep() -> None:
    seed = _prompt_seed()

    console.print(f"\n  [bold cyan]Sweep — {len(_CONFIG_MAP)} conditions, seed={seed}[/]")
    console.print("  [dim]No further input required. Running back-to-back...[/dim]\n")

    first_config = load_default_config(next(iter(_CONFIG_MAP.values())))
    if first_config.get("live_dashboard", False):
        _start_dashboard_bg()

    total = len(_CONFIG_MAP)
    for i, (cond_key, config_path) in enumerate(_CONFIG_MAP.items(), start=1):
        console.print(f"  [bold cyan]{'─' * 48}[/]")
        console.print(f"  [bold]Condition {i}/{total}:[/] {config_path}")
        config = load_default_config(config_path)
        from ridge.trainer import Trainer
        trainer = Trainer(config, seed=seed)
        trainer.train()
        console.print(f"  [bold green]✓ Condition {i}/{total} done.[/]\n")

    console.print(f"  [bold green]✓ Sweep complete — all {total} conditions finished.[/]")


def _do_launch_tensorboard(log_dir: str = "tensorboard_logs") -> None:
    from viewer.dashboard import launch_tensorboard
    proc = launch_tensorboard(log_dir)
    console.print(
        f"  [bold green]TensorBoard running at[/] http://localhost:6006  "
        f"[dim](PID {proc.pid})[/dim]"
    )
    input("  Press Enter to return to menu...")


def _do_comparison_graphs(log_dir: str = "tensorboard_logs") -> None:
    from viewer.dashboard import generate_all_plots
    out_dir = input("  Output directory [results]: ").strip() or "results"
    generate_all_plots(log_dir, out_dir)
    console.print(f"  [bold green]✓ Plots saved to[/] {out_dir}/")


def _find_latest_checkpoint() -> str | None:
    """Return path to the most recently modified .pt file under checkpoints/."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def _do_live_viewer() -> None:
    import torch
    from ridge.agent import PPOAgent
    from ridge.game import make_env
    from ridge.utils import get_device
    from viewer.live_viewer import LiveViewer

    # Auto-detect latest checkpoint
    default_ckpt = _find_latest_checkpoint()
    hint = f" [{default_ckpt}]" if default_ckpt else ""
    ckpt_path = input(f"  Checkpoint path{hint}: ").strip() or default_ckpt or ""

    if not ckpt_path or not Path(ckpt_path).exists():
        console.print("  [red]Checkpoint not found.[/red]")
        return

    device = get_device()

    # Load config from checkpoint so obs processing matches training exactly
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = raw.get("config", {})
    if not config:
        config_path = (
            input("  Config [configs/ridge_blend.yaml]: ").strip()
            or "configs/ridge_blend.yaml"
        )
        config = load_default_config(config_path)
    else:
        console.print(
            f"  [dim]Using config embedded in checkpoint "
            f"(run: {config.get('run_name', '?')})[/dim]"
        )

    env = make_env(config)
    agent = PPOAgent(config, num_actions=env.action_space.n, device=device)
    agent.load_checkpoint(ckpt_path)
    agent.network.eval()

    console.print(f"  [bold green]Loaded:[/] {ckpt_path}")
    console.print("  [dim]SPACE = pause  |  ↑↓ = speed  |  close window to exit[/dim]")

    viewer = LiveViewer(render_fps=int(config.get("render_fps", 15)))
    viewer.run_replay(env, agent, config, ckpt_path)


def _do_evaluate() -> None:
    config_path = (
        input("  Config [configs/ridge_blend.yaml]: ").strip()
        or "configs/ridge_blend.yaml"
    )
    config = load_default_config(config_path)
    ckpt_path = input("  Checkpoint path: ").strip()
    n_eps = int(input("  Episodes [10]: ").strip() or "10")
    seed = _prompt_seed()

    from scripts.evaluate import run_evaluation
    run_evaluation(config, ckpt_path, n_episodes=n_eps, seed=seed)


def _do_delete_models() -> None:
    """Wipe all checkpoints and TensorBoard logs after hard confirmation."""
    import shutil

    console.print("\n  [bold red]WARNING — This will permanently delete:[/]")
    console.print("    • [yellow]checkpoints/[/yellow]  (all saved model weights)")
    console.print("    • [yellow]tensorboard_logs/[/yellow]  (all training logs)\n")
    confirm = input("  Type DELETE to confirm, or anything else to cancel: ").strip()
    if confirm != "DELETE":
        console.print("  [dim]Cancelled.[/dim]")
        return

    deleted: list[str] = []
    for target in ("checkpoints", "tensorboard_logs"):
        p = Path(target)
        if p.exists():
            shutil.rmtree(p)
            deleted.append(target)

    if deleted:
        console.print(f"  [bold green]✓ Deleted:[/] {', '.join(deleted)}")
    else:
        console.print("  [dim]Nothing to delete — directories did not exist.[/dim]")


def _run_sharpness_sweep() -> None:
    """Menu option 11 — RQ3 blend_sharpness ablation sweep."""
    seed = _prompt_seed()

    console.print(f"\n  [bold cyan]Sharpness Sweep — {len(_SHARPNESS_MAP)} conditions, seed={seed}[/]")
    console.print("  [dim]blend_sharpness: 0.5 → 1.0 → 2.0 → 4.0. No further input required.[/dim]\n")

    first_config = load_default_config(next(iter(_SHARPNESS_MAP.values())))
    if first_config.get("live_dashboard", False):
        _start_dashboard_bg()

    total = len(_SHARPNESS_MAP)
    for i, (sharpness, config_path) in enumerate(_SHARPNESS_MAP.items(), start=1):
        console.print(f"  [bold cyan]{'─' * 48}[/]")
        console.print(f"  [bold]Sharpness {i}/{total}:[/] blend_sharpness={sharpness}  ({config_path})")
        config = load_default_config(config_path)
        from ridge.trainer import Trainer
        trainer = Trainer(config, seed=seed)
        trainer.train()
        console.print(f"  [bold green]✓ Sharpness {i}/{total} done.[/]\n")

    console.print(f"  [bold green]✓ Sharpness sweep complete — all {total} conditions finished.[/]")


def run_menu() -> None:
    """Display the RIDGE main menu and dispatch user selections."""
    setup_logging()
    while True:
        _print_menu()
        choice = input("  Select option: ").strip()

        if choice == "1":
            _run_training("configs/ridge_blend.yaml")
        elif choice == "2":
            _run_training("configs/explorer.yaml")
        elif choice == "3":
            _run_training("configs/survivor.yaml")
        elif choice == "4":
            _run_training("configs/craftsman.yaml")
        elif choice == "5":
            _run_training("configs/warrior.yaml")
        elif choice == "6":
            _run_sweep()
        elif choice == "7":
            _do_live_viewer()
        elif choice == "8":
            _do_launch_tensorboard()
        elif choice == "9":
            _do_comparison_graphs()
        elif choice == "10":
            _do_evaluate()
        elif choice == "11":
            _run_sharpness_sweep()
        elif choice.upper() == "L":
            _do_live_dashboard()
        elif choice.upper() == "D":
            _do_delete_models()
        elif choice == "0":
            _clear()
            console.print("[bold cyan]Goodbye.[/bold cyan]\n")
            break
        else:
            console.print("  [red]Invalid option.[/red]")
            input("  Press Enter to continue...")
