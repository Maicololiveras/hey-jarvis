"""Local model server management (llama-cpp-python for local-qwen)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost"
DEFAULT_PORT = 8081


class LocalModelServer:
    """Manages a local llama-cpp-python server for local-qwen backend."""

    def __init__(
        self,
        model_path: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        context_size: int = 4096,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.context_size = context_size

        self._process: subprocess.Popen | None = None
        self._started = threading.Event()
        self._stop_requested = threading.Event()
        self._url = f"{host}:{port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def url(self) -> str:
        return self._url

    def start(self, timeout: float = 30.0) -> bool:
        """Start the local model server in a background thread.

        Returns True if server started successfully or is already running.
        Returns False if server failed to start.
        """
        if self.is_running:
            log.info("[LocalModel] Server already running at %s", self._url)
            return True

        if self._started.is_set() and not self.is_running:
            log.warning("[LocalModel] Server previously failed to start")
            return False

        thread = threading.Thread(target=self._run_server, daemon=True)
        thread.start()

        if self._started.wait(timeout=timeout):
            if self.is_running:
                log.info("[LocalModel] Server started at %s", self._url)
                return True

        log.error("[LocalModel] Server failed to start within %.1fs", timeout)
        return False

    def _run_server(self) -> None:
        """Run the llama-cpp-python server (blocking)."""
        if not self.model_path:
            log.error("[LocalModel] model_path not configured")
            self._started.set()
            return

        if not os.path.exists(self.model_path):
            log.error("[LocalModel] Model file not found: %s", self.model_path)
            self._started.set()
            return

        llama_server = shutil.which("llama-server")
        if not llama_server:
            log.error("[LocalModel] llama-server not found in PATH")
            self._started.set()
            return

        cmd = [
            llama_server,
            "-m",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ngl",
            str(self.n_gpu_layers),
            "-t",
            str(self.n_threads),
            "-c",
            str(self.context_size),
            "--threads",
            str(self.n_threads),
        ]

        log.info("[LocalModel] Starting: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._started.set()

            self._process.wait()

        except Exception as exc:
            log.exception("[LocalModel] Server exception: %s", exc)
            self._started.set()

    def stop(self) -> None:
        """Stop the local model server."""
        self._stop_requested.set()

        if self._process and self.is_running:
            log.info("[LocalModel] Stopping server (PID=%d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

        self._process = None
        log.info("[LocalModel] Server stopped")


def get_server_from_config(config: dict) -> LocalModelServer | None:
    """Create a LocalModelServer from Jarvis config."""
    backends = config.get("query", {}).get("backends", {})
    local_qwen = backends.get("local-qwen", {})

    model_path = local_qwen.get("model_path", "")
    if not model_path:
        log.warning("[LocalModel] local-qwen.model_path not configured")
        return None
    if not os.path.isfile(model_path):
        log.warning("[LocalModel] Model file not found, skipping preload: %s", model_path)
        return None

    return LocalModelServer(
        model_path=model_path,
        host=local_qwen.get("host", DEFAULT_HOST),
        port=local_qwen.get("port", DEFAULT_PORT),
        n_gpu_layers=local_qwen.get("n_gpu_layers", 0),
        n_threads=local_qwen.get("n_threads", 4),
        context_size=local_qwen.get("context_size", 4096),
    )
