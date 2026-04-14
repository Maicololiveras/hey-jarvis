"""Entry point: python -m jarvis.

Architecture
------------
Qt requires the event loop on the **main thread** (macOS/Windows hard
requirement).  The audio-processing daemon blocks in its own loop, so it
runs in a background thread.

Startup sequence:
    1. ``setup_logging()`` — configure log format early.
    2. ``JarvisUI.run(on_ready=...)`` — creates QApplication + widget on
       the main thread.  The *on_ready* callback fires right before
       ``app.exec()`` and starts the daemon thread, passing the UI
       instance so both sides share the same widget.
    3. The daemon thread runs ``JarvisDaemon(ui=ui).run()`` which blocks
       until interrupted or fatal error.
"""
import sys
import threading
import logging
import tempfile
import os

from .logging_setup import setup_logging
from .jarvis_daemon import JarvisDaemon
from .jarvis_ui import JarvisUI

log = logging.getLogger(__name__)


def _start_daemon(ui: JarvisUI) -> None:
    """on_ready callback: launches the daemon in a background thread."""

    def _daemon_worker() -> None:
        try:
            daemon = JarvisDaemon(ui=ui)
            daemon.run()
        except Exception:
            log.exception("[Jarvis] Daemon thread crashed")

    t = threading.Thread(target=_daemon_worker, daemon=True, name="jarvis-daemon")
    t.start()
    log.info("[Jarvis] Daemon thread started")


_LOCK_FILE = os.path.join(tempfile.gettempdir(), "jarvis_singleton.lock")


def _check_singleton() -> bool:
    """Return True if no other Jarvis instance is running."""
    import subprocess
    try:
        result = subprocess.run(
            ["powershell.exe", "-Command",
             "Get-CimInstance Win32_Process | Where-Object "
             "{ $_.CommandLine -like '*-m jarvis*' -and $_.Name -eq 'python.exe' "
             "-and $_.ProcessId -ne " + str(os.getpid()) + " } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=5
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        if pids:
            log.warning("[Jarvis] Another instance already running (PIDs: %s). Exiting.", ", ".join(pids))
            return False
    except Exception:
        pass  # If check fails, proceed anyway
    return True


def main() -> int:
    setup_logging(level="DEBUG")
    log.info("[Jarvis] ===== Hey Jarvis starting =====")

    if not _check_singleton():
        return 0

    # Kill any orphan ffplay from previous crashed sessions
    from .tts import kill_active_playback
    kill_active_playback()

    try:
        # Qt event loop runs on the main thread (hard requirement).
        # on_ready fires once the QApplication + widget exist, spinning
        # up the daemon thread that sends UICommands back to the widget.
        JarvisUI.run(on_ready=_start_daemon)
    except KeyboardInterrupt:
        log.info("[Jarvis] Interrupted by user")
    except Exception:
        log.exception("[Jarvis] Fatal error")
        return 1

    log.info("[Jarvis] ===== Hey Jarvis stopped =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
