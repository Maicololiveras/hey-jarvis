"""Logging configuration for Jarvis daemon."""
import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent
LOG_FILE = LOG_DIR / "jarvis.log"
LOG_FILE_PREV = LOG_DIR / "jarvis.log.1"

def setup_logging(level: str = "INFO") -> None:
    """Configure logging to file with rotation on restart."""
    # Rotate previous log
    if LOG_FILE.exists():
        if LOG_FILE_PREV.exists():
            LOG_FILE_PREV.unlink()
        LOG_FILE.rename(LOG_FILE_PREV)

    # Force UTF-8 on Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    logging.getLogger("jarvis").info("Logging initialized — level=%s, file=%s", level, LOG_FILE)
