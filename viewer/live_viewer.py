"""Real-time pygame viewer for watching the RIDGE agent play Crafter."""

import logging
import time
from typing import Any

import numpy as np
import pygame

logger = logging.getLogger(__name__)

# Viewer dimensions
GAME_W, GAME_H = 512, 512
PANEL_W = 280
WIN_W = GAME_W + PANEL_W
WIN_H = GAME_H

# Colours
C_BG = (20, 20, 30)
C_WHITE = (240, 240, 240)
C_GRAY = (120, 120, 130)
C_BLACK = (0, 0, 0)
C_RED = (220, 60, 60)
C_GREEN = (60, 200, 80)
C_BLUE = (60, 120, 220)
C_YELLOW = (230, 200, 60)
C_ORANGE = (230, 130, 40)
C_PURPLE = (160, 80, 200)

PERSONA_COLOURS = [C_GREEN, C_RED, C_BLUE, C_PURPLE]  # Explorer, Survivor, Craftsman, Warrior
PERSONA_NAMES = ["Explorer", "Survivor", "Craftsman", "Warrior"]

# Speed options (multiplier on base FPS)
SPEEDS = [0.5, 1.0, 2.0, 4.0]


class LiveViewer:
    """Pygame-based live viewer with debug overlay.

    Can be used during training (push_frame class method) or
    post-training (load checkpoint and replay).
    """

    _instance: "LiveViewer | None" = None

    def __init__(self, render_fps: int = 15) -> None:
        """Initialise pygame window and state.

        Args:
            render_fps: Base frames per second for display.
        """
        pygame.init()
        self._screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("RIDGE — Live Viewer")
        self._clock = pygame.time.Clock()
        self._font_lg = pygame.font.SysFont("monospace", 16, bold=True)
        self._font_sm = pygame.font.SysFont("monospace", 13)

        self._base_fps = render_fps
        self._speed_idx = 1  # default 1×
        self._paused = False
        self._step_once = False

        self._frame: np.ndarray | None = None
        self._info: dict[str, Any] = {}
        self._weights: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self._episode_reward: float = 0.0
        self._step_count: int = 0
        self._achievement_ticker: list[tuple[str, float]] = []  # (name, expire_time)

        LiveViewer._instance = self
        logger.info("LiveViewer initialised")

    @classmethod
    def push_frame(
        cls,
        frame: np.ndarray | None,
        info: dict[str, Any],
        weights: np.ndarray,
    ) -> bool:
        """Push a new frame from the training loop (non-blocking).

        Args:
            frame: Raw RGB frame (H, W, 3) uint8 or None.
            info: Enriched info dict.
            weights: Persona weights array (4,).

        Returns:
            False if the window was closed (caller should disable live_view).
        """
        if cls._instance is None:
            return False
        cls._instance._update_state(frame, info, weights)
        cls._instance._render()
        return cls._instance._poll_events()

    def _update_state(
        self,
        frame: np.ndarray | None,
        info: dict[str, Any],
        weights: np.ndarray,
    ) -> None:
        """Update internal viewer state.

        Args:
            frame: RGB array or None.
            info: Info dict.
            weights: Persona weights (4,).
        """
        self._frame = frame
        self._info = info
        self._weights = weights
        self._step_count += 1

        # Achievement ticker
        now = time.time()
        for name, unlocked in info.get("achievements", {}).items():
            if unlocked:
                # Only add if not already in ticker
                if not any(n == name for n, _ in self._achievement_ticker):
                    self._achievement_ticker.append((name, now + 3.0))
        self._achievement_ticker = [(n, t) for n, t in self._achievement_ticker if t > now]

    def _poll_events(self) -> bool:
        """Non-blocking event poll used during training. Returns False if quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._speed_idx = min(self._speed_idx + 1, len(SPEEDS) - 1)
                elif event.key == pygame.K_DOWN:
                    self._speed_idx = max(self._speed_idx - 1, 0)
        return True

    def _handle_events(self) -> bool:
        """Process pygame events for pause, step, and speed control (replay mode).

        Returns:
            False if the window was closed, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                elif event.key == pygame.K_RIGHT:
                    self._step_once = True
                elif event.key == pygame.K_UP:
                    self._speed_idx = min(self._speed_idx + 1, len(SPEEDS) - 1)
                elif event.key == pygame.K_DOWN:
                    self._speed_idx = max(self._speed_idx - 1, 0)

        fps = self._base_fps * SPEEDS[self._speed_idx]
        self._clock.tick(fps)

        if self._paused and not self._step_once:
            while self._paused and not self._step_once:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self._paused = False
                        elif event.key == pygame.K_RIGHT:
                            self._step_once = True
                self._clock.tick(30)
        self._step_once = False
        return True

    def _render(self) -> None:
        """Draw the full window: game frame + debug panel."""
        self._screen.fill(C_BG)
        self._draw_game_frame()
        self._draw_panel()
        pygame.display.flip()

    def _draw_game_frame(self) -> None:
        """Blit the Crafter game frame to the left half of the window."""
        if self._frame is not None:
            # frame is (H, W, 3) uint8
            surf = pygame.surfarray.make_surface(np.transpose(self._frame, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (GAME_W, GAME_H))
            self._screen.blit(surf, (0, 0))
        else:
            pygame.draw.rect(self._screen, (40, 40, 50), (0, 0, GAME_W, GAME_H))
            label = self._font_lg.render("No frame", True, C_GRAY)
            self._screen.blit(label, (GAME_W // 2 - 40, GAME_H // 2))

    def _draw_panel(self) -> None:
        """Draw the debug overlay panel on the right side."""
        x0 = GAME_W + 8
        y = 10

        # Title
        self._blit(f"RIDGE Debug", x0, y, self._font_lg, C_WHITE)
        y += 24
        self._blit(f"Step: {self._step_count:,}  Speed: {SPEEDS[self._speed_idx]}x", x0, y, self._font_sm, C_GRAY)
        y += 20
        self._blit("SPACE=pause  →=step  ↑↓=speed", x0, y, self._font_sm, C_GRAY)
        y += 26

        # Persona weight bars
        self._blit("Persona Weights", x0, y, self._font_lg, C_WHITE)
        y += 20
        for i, (name, col) in enumerate(zip(PERSONA_NAMES, PERSONA_COLOURS)):
            w = float(self._weights[i])
            bar_w = int((PANEL_W - 20) * w)
            pygame.draw.rect(self._screen, col, (x0, y, bar_w, 14))
            pygame.draw.rect(self._screen, C_GRAY, (x0, y, PANEL_W - 20, 14), 1)
            self._blit(f"{name}: {w:.2f}", x0 + bar_w + 4, y, self._font_sm, col)
            y += 18
        y += 8

        # Vitals
        self._blit("Vitals", x0, y, self._font_lg, C_WHITE)
        y += 20
        vitals = [
            ("Health", self._info.get("health", 9), 9, C_RED),
            ("Food", self._info.get("food", 9), 9, C_ORANGE),
            ("Drink", self._info.get("drink", 9), 9, C_BLUE),
            ("Energy", self._info.get("energy", 9), 9, C_YELLOW),
        ]
        for label, val, max_val, col in vitals:
            frac = val / max_val
            bar_w = int((PANEL_W - 20) * frac)
            pygame.draw.rect(self._screen, col, (x0, y, bar_w, 12))
            pygame.draw.rect(self._screen, C_GRAY, (x0, y, PANEL_W - 20, 12), 1)
            self._blit(f"{label}: {val}/{max_val}", x0 + bar_w + 4, y, self._font_sm, col)
            y += 16
        y += 8

        # Inventory (top items only)
        self._blit("Inventory", x0, y, self._font_lg, C_WHITE)
        y += 20
        inv = self._info.get("inventory", {})
        inv_items = [(k, v) for k, v in inv.items() if v > 0]
        for k, v in inv_items[:8]:
            self._blit(f"  {k}: {v}", x0, y, self._font_sm, C_WHITE)
            y += 15
        y += 8

        # Achievement ticker
        self._blit("Achievements", x0, y, self._font_lg, C_WHITE)
        y += 20
        now = time.time()
        for name, expire in self._achievement_ticker[-4:]:
            remaining = expire - now
            alpha = min(1.0, remaining)
            colour = tuple(int(c * alpha) for c in C_YELLOW)
            self._blit(f"  ✓ {name}", x0, y, self._font_sm, colour)
            y += 15

        # FPS / paused indicator
        if self._paused:
            label = self._font_lg.render("[ PAUSED ]", True, C_YELLOW)
            self._screen.blit(label, (GAME_W // 2 - 50, GAME_H - 30))

    def _blit(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        colour: tuple[int, int, int],
    ) -> None:
        surf = font.render(text, True, colour)
        self._screen.blit(surf, (x, y))

    def close(self) -> None:
        """Shut down the pygame window."""
        pygame.quit()
        LiveViewer._instance = None
        logger.info("LiveViewer closed")

    def run_replay(
        self,
        env: Any,
        agent: Any,
        config: dict[str, Any],
        checkpoint_path: str,
    ) -> None:
        """Run a post-training replay from a checkpoint.

        Args:
            env: CrafterWrapper instance.
            agent: PPOAgent instance (checkpoint already loaded).
            config: Config dict.
            checkpoint_path: Path to the checkpoint (already loaded externally).
        """
        from ridge.rewards import compute_blended_reward
        import numpy as np

        def _fresh_unlocked() -> dict:
            return {"explorer": set(), "survivor": set(), "craftsman": set(), "warrior": set()}

        obs, info = env.reset()
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        episode_unlocked = _fresh_unlocked()

        while True:
            state_vec = env.extract_state_vector(info)
            _, weights, _ = compute_blended_reward(info, state_vec, config, episode_unlocked)
            action, _, _, _ = agent.select_action(obs, weights)
            obs, _, terminated, truncated, info = env.step(action)

            frame = env.render()
            self._update_state(frame, info, weights)
            self._render()
            if not self._handle_events():
                break

            if terminated or truncated:
                obs, info = env.reset()
                weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
                episode_unlocked = _fresh_unlocked()

        if LiveViewer._instance is not None:
            self.close()
