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


def main() -> int:
    setup_logging(level="DEBUG")
    log.info("[Jarvis] ===== Hey Jarvis starting =====")

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
