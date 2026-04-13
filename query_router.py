"""Query routing to AI backends."""
from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

from .models import QueryResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
SYSTEM_PROMPT_LOCAL_FILE = Path(__file__).parent / "system_prompt_local.txt"


class QueryRouter:
    """Routes queries to the appropriate AI backend."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._system_prompt = self._load_prompt(SYSTEM_PROMPT_FILE)
        self._system_prompt_local = self._load_prompt(SYSTEM_PROMPT_LOCAL_FILE)
        self._default_backend = config.get("default_backend", "claude-p")
        self._timeout = config.get("timeout_seconds", 60)
        log.info("[QueryRouter] Initialized — default=%s, timeout=%ds", self._default_backend, self._timeout)

    @staticmethod
    def _load_prompt(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            log.warning("System prompt not found: %s", path)
            return ""

    # Ordered fallback chains per backend.
    _FALLBACK_ORDER: dict[str, list[str]] = {
        "claude-p": ["claude-p", "local-qwen"],
        "local-qwen": ["local-qwen", "claude-p"],
    }

    def _dispatch(self, backend: str, user_text: str) -> QueryResult:
        """Dispatch a query to a single backend by name."""
        if backend == "claude-p":
            return self._query_claude_p(user_text)
        if backend == "local-qwen":
            return self._query_local_qwen(user_text)
        return QueryResult(ok=False, error=f"Unknown backend: {backend}")

    def query(self, user_text: str, backend: str | None = None) -> QueryResult:
        """Send a query to the specified backend with automatic fallback."""
        backend = backend or self._default_backend
        chain = self._FALLBACK_ORDER.get(backend, [backend])
        log.info("[QueryRouter] Querying backend=%s, text=%s...", backend, user_text[:80])

        last_result: QueryResult | None = None
        for attempt_backend in chain:
            start = time.time()
            result = self._dispatch(attempt_backend, user_text)
            latency = (time.time() - start) * 1000

            if result.ok:
                return QueryResult(
                    ok=True,
                    text=result.text,
                    backend=attempt_backend,
                    latency_ms=latency,
                    error=None,
                )

            last_result = QueryResult(
                ok=False,
                text=result.text,
                backend=attempt_backend,
                latency_ms=latency,
                error=result.error,
            )

            # If there are more backends in the chain, log the fallback.
            remaining = chain[chain.index(attempt_backend) + 1:]
            if remaining:
                next_backend = remaining[0]
                log.warning(
                    "[QueryRouter] Backend %s failed, falling back to %s",
                    attempt_backend,
                    next_backend,
                )

        # All backends exhausted — return the last error.
        return last_result  # type: ignore[return-value]

    def _query_claude_p(self, user_text: str) -> QueryResult:
        """Query via claude -p subprocess."""
        prompt = f"{self._system_prompt}\n\n---\n\nUser: {user_text}"
        backends_cfg = self._config.get("backends", {}).get("claude-p", {})
        command = backends_cfg.get("command", "claude")
        args = backends_cfg.get("args", ["-p"])

        try:
            result = subprocess.run(
                [command, *args, prompt],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip()[:200] if result.stderr else f"exit code {result.returncode}"
                log.error("[QueryRouter] claude-p failed: %s", error_msg)
                return QueryResult(ok=False, error=error_msg)

            response = result.stdout.strip()
            log.info("[QueryRouter] claude-p responded (%d chars)", len(response))
            return QueryResult(ok=True, text=response)

        except FileNotFoundError:
            log.error("[QueryRouter] claude binary not found — is claude-code installed?")
            return QueryResult(ok=False, error="claude binary not found")
        except subprocess.TimeoutExpired:
            log.error("[QueryRouter] claude-p timed out after %ds", self._timeout)
            return QueryResult(ok=False, error=f"timeout after {self._timeout}s")

    def _query_local_qwen(self, user_text: str) -> QueryResult:
        """Query local Qwen via llama-cpp-python OpenAI-compatible API."""
        backends_cfg = self._config.get("backends", {}).get("local-qwen", {})
        host = backends_cfg.get("host", "http://localhost")
        port = backends_cfg.get("port", 8081)
        url = f"{host}:{port}/v1/chat/completions"

        payload = json.dumps({
            "messages": [
                {"role": "system", "content": self._system_prompt_local},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            choices = body.get("choices", [])
            if not choices:
                log.error("[QueryRouter] local-qwen returned no choices")
                return QueryResult(ok=False, error="local-qwen returned empty choices")

            text = choices[0].get("message", {}).get("content", "").strip()
            log.info("[QueryRouter] local-qwen responded (%d chars)", len(text))
            return QueryResult(ok=True, text=text)

        except ConnectionRefusedError:
            log.error("[QueryRouter] local-qwen server not running at %s", url)
            return QueryResult(ok=False, error=f"local-qwen server not running at {url}")
        except urllib.error.URLError as exc:
            log.error("[QueryRouter] local-qwen connection error: %s", exc.reason)
            return QueryResult(ok=False, error=f"local-qwen connection error: {exc.reason}")
        except urllib.error.HTTPError as exc:
            log.error("[QueryRouter] local-qwen HTTP %d: %s", exc.code, exc.reason)
            return QueryResult(ok=False, error=f"local-qwen HTTP {exc.code}: {exc.reason}")
        except TimeoutError:
            log.error("[QueryRouter] local-qwen timed out after %ds", self._timeout)
            return QueryResult(ok=False, error=f"local-qwen timeout after {self._timeout}s")
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            log.error("[QueryRouter] local-qwen response parse error: %s", exc)
            return QueryResult(ok=False, error=f"local-qwen response parse error: {exc}")
