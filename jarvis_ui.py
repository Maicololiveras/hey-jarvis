"""Jarvis visual UI — circular transparent window with audio waveform."""
from __future__ import annotations

import logging
import math
import queue
import sys
import time
from typing import Optional

import numpy as np

from .models import UICommand

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import PyQt6 — handle missing gracefully
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtCore import Qt, QTimer, QRect, QPointF
    from PyQt6.QtGui import (
        QPainter,
        QColor,
        QRadialGradient,
        QBrush,
        QPen,
        QRegion,
    )
    from PyQt6.QtWidgets import QApplication, QWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    log.warning("PyQt6 not installed — UI disabled")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_ui_config() -> dict:
    """Load UI section from config, with sane defaults."""
    try:
        from .config import get_ui_config
        return get_ui_config()
    except Exception:
        return {}


def _parse_rgba(rgba_str: str) -> tuple[int, int, int, int]:
    """Parse 'rgba(r, g, b, a)' into (r, g, b, alpha_0_255)."""
    try:
        inner = rgba_str.strip().removeprefix("rgba(").removesuffix(")")
        parts = [p.strip() for p in inner.split(",")]
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        a = int(float(parts[3]) * 255)
        return (r, g, b, a)
    except Exception:
        return (0, 20, 40, 153)  # fallback


# ---------------------------------------------------------------------------
# JarvisUI — the real implementation (requires PyQt6)
# ---------------------------------------------------------------------------

if HAS_PYQT6:

    class JarvisUI(QWidget):  # type: ignore[misc]
        """Circular transparent always-on-top window for Jarvis visual feedback."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)

            # --- Config -------------------------------------------------
            cfg = _load_ui_config()
            self._diameter: int = cfg.get("diameter", 200)
            colors = cfg.get("colors", {})
            self._color_primary = QColor(colors.get("primary", "#00BFFF"))
            self._color_glow = QColor(colors.get("glow", "#FFFFFF"))
            bg = colors.get("background", "rgba(0, 20, 40, 0.6)")
            r, g, b, a = _parse_rgba(bg)
            self._color_bg = QColor(r, g, b, a)
            fps = cfg.get("fps", 60)
            timer_interval_ms = max(1, 1000 // fps)  # ~16ms for 60fps

            # --- UI state ----------------------------------------------
            self._ui_state: str = "idle"
            self._state_time: float = time.time()

            # --- Waveform state ----------------------------------------
            self._num_lines: int = 48
            self._waveform_data: np.ndarray = np.zeros(self._num_lines)
            self._idle_phase: float = 0.0  # for idle pulsing animation

            # --- State color map ---------------------------------------
            self._state_colors = {
                "idle":       QColor("#00BFFF"),
                "listening":  QColor("#00BFFF"),
                "processing": QColor("#00D4E8"),  # warmer cyan
                "speaking":   QColor("#00FFFF"),   # cyan-green
            }

            # --- Thread-safe command queue ------------------------------
            self._queue: queue.Queue[UICommand] = queue.Queue()

            # --- Window flags ------------------------------------------
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setFixedSize(self._diameter, self._diameter)

            # --- Circular mask -----------------------------------------
            self._apply_mask()

            # --- Position: center of primary screen --------------------
            self._center_on_screen(cfg.get("position", "center"))

            # --- Poll timer --------------------------------------------
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._process_queue)
            self._timer.start(timer_interval_ms)

            log.info(
                "JarvisUI initialized: %dpx circle, %dfps timer",
                self._diameter,
                fps,
            )

        # ---------------------------------------------------------------
        # Geometry helpers
        # ---------------------------------------------------------------

        def _apply_mask(self) -> None:
            """Set an elliptical mask so only the circle is clickable/visible."""
            region = QRegion(
                QRect(0, 0, self._diameter, self._diameter),
                QRegion.RegionType.Ellipse,
            )
            self.setMask(region)

        def _center_on_screen(self, position: str) -> None:
            """Move the widget to the center of the primary screen."""
            app = QApplication.instance()
            if app is None:
                return
            screen = app.primaryScreen()  # type: ignore[union-attr]
            if screen is None:
                return
            geo = screen.availableGeometry()
            if position == "center":
                x = geo.x() + (geo.width() - self._diameter) // 2
                y = geo.y() + (geo.height() - self._diameter) // 2
            else:
                # Default to center for any unrecognized position value
                x = geo.x() + (geo.width() - self._diameter) // 2
                y = geo.y() + (geo.height() - self._diameter) // 2
            self.move(x, y)

        # ---------------------------------------------------------------
        # State management
        # ---------------------------------------------------------------

        def set_state(self, state: str) -> None:
            """Change the UI animation state.

            Valid states: 'idle', 'listening', 'processing', 'speaking'.
            """
            valid = {"idle", "listening", "processing", "speaking"}
            if state not in valid:
                log.warning("set_state: unknown state %r (valid: %s)", state, valid)
                return
            if state != self._ui_state:
                log.debug("UI state: %s -> %s", self._ui_state, state)
                self._ui_state = state
                self._state_time = time.time()
                self._color_primary = self._state_colors.get(state, self._state_colors["idle"])
                self.update()

        # ---------------------------------------------------------------
        # Paint
        # ---------------------------------------------------------------

        def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
            """Draw the filled circle with radial waveform lines."""
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            center_x = self._diameter / 2.0
            center_y = self._diameter / 2.0
            radius = self._diameter / 2.0

            # --- Outer glow (radial gradient behind the border) --------
            glow_margin = 6
            glow_gradient = QRadialGradient(center_x, center_y, radius)
            glow_color = QColor(self._color_primary)
            glow_color.setAlpha(80)
            glow_gradient.setColorAt(0.0, QColor(0, 0, 0, 0))
            glow_gradient.setColorAt(0.85, QColor(0, 0, 0, 0))
            glow_gradient.setColorAt(0.92, glow_color)
            glow_gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow_gradient))
            painter.drawEllipse(0, 0, self._diameter, self._diameter)

            # --- Background fill ---------------------------------------
            inset = glow_margin
            inner_d = self._diameter - inset * 2
            painter.setBrush(QBrush(self._color_bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(inset, inset, inner_d, inner_d)

            # --- Border ring -------------------------------------------
            pen = QPen(self._color_primary, 2.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            border_offset = glow_margin + 1
            border_d = self._diameter - border_offset * 2
            painter.drawEllipse(border_offset, border_offset, border_d, border_d)

            # --- Center glow gradient ----------------------------------
            center_glow = QRadialGradient(center_x, center_y, radius * 0.35)
            center_glow.setColorAt(0.0, QColor(255, 255, 255, 40))
            center_glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(center_glow))
            glow_r = radius * 0.35
            painter.drawEllipse(
                QPointF(center_x, center_y), glow_r, glow_r
            )

            # --- Radial waveform lines (state-driven) ------------------
            has_audio = np.any(self._waveform_data > 0.01)
            state = self._ui_state
            elapsed = time.time() - self._state_time

            # Inner radius where lines start (just outside center glow)
            inner_r = radius * 0.25
            # Max outer extent for line tips
            max_line_len = radius * 0.55

            # Processing state: breathing circle instead of waveform lines
            if state == "processing":
                # Slow sine breathing at ~0.5Hz
                breath = 0.5 + 0.15 * math.sin(2.0 * math.pi * 0.5 * elapsed)
                breath_r = radius * breath

                # Pulsing glow ring
                glow_alpha = int(40 + 30 * math.sin(2.0 * math.pi * 0.5 * elapsed))
                proc_glow = QRadialGradient(center_x, center_y, breath_r * 1.3)
                proc_color = QColor(self._color_primary)
                proc_color.setAlpha(glow_alpha)
                proc_glow.setColorAt(0.0, QColor(0, 0, 0, 0))
                proc_glow.setColorAt(0.7, proc_color)
                proc_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(proc_glow))
                painter.drawEllipse(
                    QPointF(center_x, center_y), breath_r * 1.3, breath_r * 1.3
                )

                # Breathing ring outline
                ring_alpha = int(100 + 80 * math.sin(2.0 * math.pi * 0.5 * elapsed))
                ring_color = QColor(self._color_primary)
                ring_color.setAlpha(ring_alpha)
                ring_pen = QPen(ring_color, 2.5)
                ring_pen.setCosmetic(True)
                painter.setPen(ring_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(
                    QPointF(center_x, center_y), breath_r, breath_r
                )

                # Still draw subtle short waveform lines for texture
                for i in range(self._num_lines):
                    angle = (2.0 * math.pi * i) / self._num_lines
                    amp = 0.05 + 0.03 * math.sin(
                        2.0 * math.pi * 0.5 * elapsed + i * 0.3
                    )
                    line_len = inner_r + max_line_len * amp
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    x1 = center_x + inner_r * cos_a
                    y1 = center_y + inner_r * sin_a
                    x2 = center_x + line_len * cos_a
                    y2 = center_y + line_len * sin_a
                    lc = QColor(self._color_primary)
                    lc.setAlpha(int(60 + 40 * amp))
                    lp = QPen(lc, 1.5)
                    lp.setCosmetic(True)
                    painter.setPen(lp)
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

                painter.end()
                self.update()  # keep processing animation ticking
                return

            # --- States: idle, listening, speaking — radial waveform ---
            for i in range(self._num_lines):
                angle = (2.0 * math.pi * i) / self._num_lines

                if state == "idle":
                    # Very subtle ambient pulse — barely perceptible, dimmer
                    self._idle_phase += 0.0002
                    amp = 0.04 + 0.03 * math.sin(
                        self._idle_phase + i * (2.0 * math.pi / self._num_lines) * 2
                    )
                elif state in ("listening", "speaking") and has_audio:
                    amp = float(self._waveform_data[i % len(self._waveform_data)])
                else:
                    # Fallback idle-like animation when no audio in listening/speaking
                    self._idle_phase += 0.0003
                    amp = 0.08 + 0.06 * math.sin(
                        self._idle_phase + i * (2.0 * math.pi / self._num_lines) * 3
                    )

                line_len = inner_r + max_line_len * min(amp, 1.0)

                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                x1 = center_x + inner_r * cos_a
                y1 = center_y + inner_r * sin_a
                x2 = center_x + line_len * cos_a
                y2 = center_y + line_len * sin_a

                # Alpha: idle is dimmer, active states are brighter
                if state == "idle":
                    alpha = int(60 + 80 * min(amp, 1.0))
                else:
                    alpha = int(120 + 135 * min(amp, 1.0))

                line_color = QColor(self._color_primary)
                line_color.setAlpha(alpha)
                line_pen = QPen(line_color, 2.0)
                line_pen.setCosmetic(True)
                painter.setPen(line_pen)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            painter.end()

            # Keep animation ticking for non-audio states
            if state == "idle" or not has_audio:
                self.update()

        # ---------------------------------------------------------------
        # Command queue
        # ---------------------------------------------------------------

        def send_command(self, cmd: UICommand) -> None:
            """Thread-safe: enqueue a UICommand from any thread."""
            self._queue.put(cmd)

        def _process_queue(self) -> None:
            """Called by QTimer on the GUI thread — drain the queue."""
            processed = 0
            while not self._queue.empty():
                try:
                    cmd = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._handle_command(cmd)
                processed += 1
            # Repaint if anything was processed
            if processed:
                self.update()

        def update_waveform(self, audio_samples: np.ndarray) -> None:
            """Compute FFT of *audio_samples* and store frequency bins for rendering.

            Parameters
            ----------
            audio_samples:
                Raw audio buffer (mono, any sample rate). The FFT magnitude
                is binned into ``self._num_lines`` frequency bands.
            """
            if audio_samples is None or len(audio_samples) == 0:
                self._waveform_data = np.zeros(self._num_lines)
                return

            # Compute FFT magnitudes (positive frequencies only)
            fft_vals = np.abs(np.fft.rfft(audio_samples.astype(np.float32)))

            # Bin into _num_lines bands by averaging
            n_bins = len(fft_vals)
            bin_size = max(1, n_bins // self._num_lines)
            bands = np.zeros(self._num_lines)
            for i in range(self._num_lines):
                start = i * bin_size
                end = min(start + bin_size, n_bins)
                if start < n_bins:
                    bands[i] = np.mean(fft_vals[start:end])

            # Normalize to 0..1
            peak = bands.max()
            if peak > 0:
                bands /= peak

            self._waveform_data = bands

        def _handle_command(self, cmd: UICommand) -> None:
            """Dispatch a single UICommand."""
            action = cmd.action
            if action == "show":
                self.show_ui()
            elif action == "hide":
                self.hide_ui()
            elif action == "update_waveform":
                data = cmd.data if cmd.data is not None else np.array([])
                if isinstance(data, np.ndarray):
                    self.update_waveform(data)
                else:
                    log.warning("update_waveform: expected np.ndarray, got %s", type(data))
            elif action == "set_state":
                state = cmd.data if isinstance(cmd.data, str) else "idle"
                self.set_state(state)
            else:
                log.warning("Unknown UICommand action: %s", action)

        # ---------------------------------------------------------------
        # Public API
        # ---------------------------------------------------------------

        def show_ui(self) -> None:
            """Show the circular window."""
            self.show()
            self.raise_()
            log.debug("JarvisUI shown")

        def hide_ui(self) -> None:
            """Hide the circular window."""
            self.hide()
            log.debug("JarvisUI hidden")

        @staticmethod
        def run(on_ready=None) -> None:
            """Create QApplication and start the event loop (blocking).

            Must be called from the **main thread** (Qt requirement on most
            platforms).  Blocks until the app quits.

            Parameters
            ----------
            on_ready:
                Optional callback ``fn(ui)`` invoked with the JarvisUI
                instance right before entering ``app.exec()``.  Useful for
                passing the instance to other threads (e.g. the daemon).
            """
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

            ui = JarvisUI()
            # Start hidden — daemon sends UICommand("show") on wake word
            ui.hide()

            if on_ready is not None:
                on_ready(ui)

            app.exec()

else:
    # ------------------------------------------------------------------
    # NullUI — fallback when PyQt6 is not available
    # ------------------------------------------------------------------

    class JarvisUI:  # type: ignore[no-redef]
        """No-op UI fallback when PyQt6 is not installed."""

        def __init__(self) -> None:
            log.info("NullUI: PyQt6 not available — all UI calls are no-ops")
            self._queue: queue.Queue[UICommand] = queue.Queue()

        def send_command(self, cmd: UICommand) -> None:
            log.debug("NullUI ignoring command: %s", cmd.action)

        def set_state(self, state: str) -> None:
            log.debug("NullUI.set_state(%s) — no-op", state)

        def show_ui(self) -> None:
            log.debug("NullUI.show_ui() — no-op")

        def hide_ui(self) -> None:
            log.debug("NullUI.hide_ui() — no-op")

        @staticmethod
        def run(on_ready=None) -> None:
            log.warning("NullUI.run() called — PyQt6 not installed, nothing to show")
            if on_ready is not None:
                on_ready(JarvisUI())
