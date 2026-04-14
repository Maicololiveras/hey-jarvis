"""Query routing to AI backends."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

from .models import QueryResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
SYSTEM_PROMPT_LOCAL_FILE = Path(__file__).parent / "system_prompt_local.txt"
DEFAULT_CLAUDE_P_ARGS = ["-p", "--bare", "--model", "haiku", "--no-session-persistence"]
CLAUDE_P_INVALID_RESPONSE_PATTERNS = (
    "session terminated.",
    "session terminated",
)


class QueryRouter:
    """Routes queries to the appropriate AI backend."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._system_prompt = self._load_prompt(SYSTEM_PROMPT_FILE)
        self._system_prompt_local = self._load_prompt(SYSTEM_PROMPT_LOCAL_FILE)
        self._default_backend = config.get("default_backend", "claude-p")
        self._timeout = config.get("timeout_seconds", 60)
        self._max_history = self._coerce_max_history(config.get("max_history", 5))
        self._history: list[dict[str, str]] = []
        self._lock = threading.Lock()
        log.info(
            "[QueryRouter] Initialized — default=%s, timeout=%ds, max_history=%d",
            self._default_backend,
            self._timeout,
            self._max_history,
        )

    @staticmethod
    def _load_prompt(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            log.warning("System prompt not found: %s", path)
            return ""

    @staticmethod
    def _coerce_max_history(value: object) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 5

    # Ordered fallback chains per backend.
    _FALLBACK_ORDER: dict[str, list[str]] = {
        "claude-p": ["claude-p", "opencode", "local-qwen"],
        "claude-api": ["claude-api", "opencode", "local-qwen"],
        "opencode": ["opencode", "claude-p", "local-qwen"],
        "openai": ["openai", "local-qwen"],
        "gemini": ["gemini", "local-qwen"],
        "local-qwen": ["local-qwen", "opencode", "claude-p"],
    }

    def _dispatch(self, backend: str, user_text: str) -> QueryResult:
        """Dispatch a query to a single backend by name."""
        if backend == "claude-p":
            return self._query_claude_p(user_text)
        if backend == "claude-api":
            return self._query_claude_api(user_text)
        if backend == "opencode":
            return self._query_opencode(user_text)
        if backend == "openai":
            return self._query_openai(user_text)
        if backend == "gemini":
            return self._query_gemini(user_text)
        if backend == "local-qwen":
            return self._query_local_qwen(user_text)
        return QueryResult(ok=False, error=f"Unknown backend: {backend}")

    @staticmethod
    def _resolve_command(command: str) -> str:
        """Resolve CLI shims reliably on Windows before spawning subprocesses."""
        resolved = shutil.which(command)
        if resolved:
            return resolved

        if os.name == "nt":
            for suffix in (".cmd", ".exe", ".bat"):
                resolved = shutil.which(f"{command}{suffix}")
                if resolved:
                    return resolved

        return command

    def _history_messages(self) -> list[dict[str, str]]:
        if self._max_history <= 0:
            return []
        return self._history[-(self._max_history * 2) :]

    def _chat_messages(
        self, user_text: str, system_prompt: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._history_messages())
        messages.append({"role": "user", "content": user_text})
        return messages

    def _conversation_messages(self, user_text: str) -> list[dict[str, str]]:
        messages = self._history_messages()
        messages.append({"role": "user", "content": user_text})
        return messages

    def _prompt_with_history(self, user_text: str, system_prompt: str) -> str:
        history_lines: list[str] = []
        for message in self._history_messages():
            role = message.get("role", "user").capitalize()
            content = message.get("content", "").strip()
            if content:
                history_lines.append(f"{role}: {content}")

        if history_lines:
            history_block = "\n".join(history_lines)
            return (
                f"{system_prompt}\n\n---\n\nConversation history:\n{history_block}"
                f"\n\nUser: {user_text}"
            )

        return f"{system_prompt}\n\n---\n\nUser: {user_text}"

    def clear_history(self) -> None:
        """Clear conversation history (e.g. on deactivation)."""
        self._history.clear()
        log.debug("[QueryRouter] Conversation history cleared")

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        """Manually append a user/assistant exchange to history."""
        self._record_history(user_text, assistant_text)

    def _record_history(self, user_text: str, response_text: str) -> None:
        if self._max_history <= 0:
            self._history.clear()
            return

        self._history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response_text},
            ]
        )
        self._history = self._history[-(self._max_history * 2) :]

    def clear_history(self) -> None:
        """Clear conversation history (called on deactivate)."""
        self._history.clear()
        log.info("[QueryRouter] Conversation history cleared")

    def query(self, user_text: str, backend: str | None = None) -> QueryResult:
        """Send a query to the specified backend with automatic fallback."""
        if backend is None:
            with self._lock:
                backend = self._default_backend
        chain = self._FALLBACK_ORDER.get(backend, [backend])
        log.info(
            "[QueryRouter] Querying backend=%s, text=%s...", backend, user_text[:80]
        )

        last_result: QueryResult | None = None
        for attempt_backend in chain:
            start = time.time()
            result = self._dispatch(attempt_backend, user_text)
            latency = (time.time() - start) * 1000

            if result.ok:
                self._record_history(user_text, result.text)
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
            remaining = chain[chain.index(attempt_backend) + 1 :]
            if remaining:
                next_backend = remaining[0]
                log.warning(
                    "[QueryRouter] Backend %s failed, falling back to %s",
                    attempt_backend,
                    next_backend,
                )

        # All backends exhausted — return the last error.
        return last_result  # type: ignore[return-value]

    @staticmethod
    def _normalize_claude_p_args(args: object) -> list[str]:
        """Normalize Claude CLI args while respecting runtime config."""
        normalized = (
            [str(arg) for arg in args if str(arg).strip()]
            if isinstance(args, list)
            else DEFAULT_CLAUDE_P_ARGS.copy()
        )
        if not normalized:
            normalized = DEFAULT_CLAUDE_P_ARGS.copy()
        if "-p" not in normalized:
            normalized.insert(0, "-p")
        return normalized

    @staticmethod
    def _is_invalid_claude_p_response(response: str) -> bool:
        normalized = response.strip()
        if not normalized:
            return True
        # Strip trailing session-terminated noise from claude -p output
        for pattern in CLAUDE_P_INVALID_RESPONSE_PATTERNS:
            if normalized.lower().endswith(pattern):
                normalized = normalized[: -len(pattern)].strip()
        # If after stripping noise there's still real content, it's valid
        if len(normalized) > 10:
            return False
        # Only reject if the ENTIRE response is garbage
        return normalized.lower() in CLAUDE_P_INVALID_RESPONSE_PATTERNS or not normalized

    def get_default_backend(self) -> str:
        with self._lock:
            return self._default_backend

    def set_default_backend(self, backend: str) -> None:
        available = self.get_available_backends()
        if backend not in available:
            raise ValueError(f"Unknown backend: {backend}")
        with self._lock:
            self._default_backend = backend
        log.info("[QueryRouter] Default backend changed to %s", backend)

    def get_available_backends(self) -> list[str]:
        configured = list(self._config.get("backends", {}).keys())
        return [
            backend for backend in configured if backend in self._dispatchable_backends
        ]

    @property
    def _dispatchable_backends(self) -> set[str]:
        return {
            "claude-p",
            "claude-api",
            "opencode",
            "openai",
            "gemini",
            "local-qwen",
        }

    def _query_claude_p(self, user_text: str) -> QueryResult:
        """Query via claude -p subprocess."""
        prompt = self._prompt_with_history(user_text, self._system_prompt)
        backends_cfg = self._config.get("backends", {}).get("claude-p", {})
        command = self._resolve_command(backends_cfg.get("command", "claude"))
        args = self._normalize_claude_p_args(backends_cfg.get("args", ["-p"]))

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
                error_msg = (
                    result.stderr.strip()[:200]
                    if result.stderr
                    else f"exit code {result.returncode}"
                )
                log.error("[QueryRouter] claude-p failed: %s", error_msg)
                return QueryResult(ok=False, error=error_msg)

            response = result.stdout.strip()
            if self._is_invalid_claude_p_response(response):
                log.warning(
                    "[QueryRouter] claude-p returned invalid response: %r",
                    response[:200],
                )
                return QueryResult(ok=False, error="claude-p returned invalid response")
            log.info("[QueryRouter] claude-p responded (%d chars)", len(response))
            return QueryResult(ok=True, text=response)

        except FileNotFoundError:
            log.error(
                "[QueryRouter] claude binary not found — is claude-code installed?"
            )
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

        payload = json.dumps(
            {
                "messages": self._chat_messages(user_text, self._system_prompt_local),
                "temperature": 0.7,
                "max_tokens": 512,
            }
        ).encode("utf-8")

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
            return QueryResult(
                ok=False, error=f"local-qwen server not running at {url}"
            )
        except urllib.error.URLError as exc:
            log.error("[QueryRouter] local-qwen connection error: %s", exc.reason)
            return QueryResult(
                ok=False, error=f"local-qwen connection error: {exc.reason}"
            )
        except urllib.error.HTTPError as exc:
            log.error("[QueryRouter] local-qwen HTTP %d: %s", exc.code, exc.reason)
            return QueryResult(
                ok=False, error=f"local-qwen HTTP {exc.code}: {exc.reason}"
            )
        except TimeoutError:
            log.error("[QueryRouter] local-qwen timed out after %ds", self._timeout)
            return QueryResult(
                ok=False, error=f"local-qwen timeout after {self._timeout}s"
            )
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            log.error("[QueryRouter] local-qwen response parse error: %s", exc)
            return QueryResult(
                ok=False, error=f"local-qwen response parse error: {exc}"
            )

    def _query_opencode(self, user_text: str) -> QueryResult:
        """Query OpenCode via non-interactive JSON event stream."""
        prompt = self._prompt_with_history(user_text, self._system_prompt)
        backend_cfg = self._get_backend_config("opencode")
        command = self._resolve_command(backend_cfg.get("command", "opencode"))
        args = backend_cfg.get("args", ["run", "--format", "json"])

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
                error_msg = (
                    result.stderr.strip()[:200]
                    if result.stderr
                    else f"exit code {result.returncode}"
                )
                log.error("[QueryRouter] opencode failed: %s", error_msg)
                return QueryResult(ok=False, error=error_msg)

            text = self._extract_opencode_text(result.stdout)
            if not text:
                return QueryResult(
                    ok=False, error="opencode returned empty text stream"
                )

            log.info("[QueryRouter] opencode responded (%d chars)", len(text))
            return QueryResult(ok=True, text=text)

        except FileNotFoundError:
            log.error("[QueryRouter] opencode binary not found")
            return QueryResult(ok=False, error="opencode binary not found")
        except subprocess.TimeoutExpired:
            log.error("[QueryRouter] opencode timed out after %ds", self._timeout)
            return QueryResult(ok=False, error=f"timeout after {self._timeout}s")

    @staticmethod
    def _extract_opencode_text(stdout: str) -> str:
        """Extract the final text payload from OpenCode JSONL output."""
        parts: list[str] = []
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "text":
                continue

            candidate = ""
            part = event.get("part")
            if isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    candidate = part_text

            if not candidate:
                text = event.get("text")
                if isinstance(text, str) and text.strip():
                    candidate = text

            if not candidate:
                candidate = QueryRouter._extract_text_content(event.get("content"))

            if candidate:
                parts.append(candidate)

        return "".join(parts).strip()

    @staticmethod
    def _extract_text_content(content: object) -> str:
        """Normalize SDK-specific content payloads to plain text."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part.strip() for part in parts if part).strip()
        return ""

    def _get_backend_config(self, backend: str) -> dict:
        return self._config.get("backends", {}).get(backend, {})

    def _get_api_key(self, backend: str, default_env: str) -> tuple[str | None, str]:
        backend_cfg = self._get_backend_config(backend)
        env_name = backend_cfg.get("api_key_env", default_env)
        api_key = os.getenv(env_name, "").strip()
        if not api_key:
            return None, env_name
        return api_key, env_name

    def _query_openai(self, user_text: str) -> QueryResult:
        """Query OpenAI directly via the official SDK."""
        api_key, env_name = self._get_api_key("openai", "OPENAI_API_KEY")
        if not api_key:
            return QueryResult(ok=False, error=f"Missing API key in env var {env_name}")

        try:
            from openai import OpenAI  # noqa: WPS433 (lazy import)
        except ImportError:
            return QueryResult(
                ok=False, error="OpenAI SDK not installed; add package 'openai'"
            )

        backend_cfg = self._get_backend_config("openai")
        model = backend_cfg.get("model", "gpt-4o")

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=self._chat_messages(user_text, self._system_prompt),
                timeout=self._timeout,
            )
            text = response.choices[0].message.content or ""
            return QueryResult(ok=True, text=text.strip())
        except Exception as exc:  # pragma: no cover - network/sdk surface
            log.error("[QueryRouter] openai failed: %s", exc)
            return QueryResult(ok=False, error=f"openai request failed: {exc}")

    def _query_claude_api(self, user_text: str) -> QueryResult:
        """Query Anthropic directly via the official SDK."""
        api_key, env_name = self._get_api_key("claude-api", "ANTHROPIC_API_KEY")
        if not api_key:
            return QueryResult(ok=False, error=f"Missing API key in env var {env_name}")

        try:
            from anthropic import Anthropic  # noqa: WPS433 (lazy import)
        except ImportError:
            return QueryResult(
                ok=False, error="Anthropic SDK not installed; add package 'anthropic'"
            )

        backend_cfg = self._get_backend_config("claude-api")
        model = backend_cfg.get("model", "claude-sonnet-4-6")

        try:
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=self._system_prompt,
                messages=self._conversation_messages(user_text),
                timeout=self._timeout,
            )
            text = self._extract_text_content(getattr(response, "content", []))
            if not text:
                return QueryResult(ok=False, error="claude-api returned empty content")
            return QueryResult(ok=True, text=text)
        except Exception as exc:  # pragma: no cover - network/sdk surface
            log.error("[QueryRouter] claude-api failed: %s", exc)
            return QueryResult(ok=False, error=f"claude-api request failed: {exc}")

    def _query_gemini(self, user_text: str) -> QueryResult:
        """Query Gemini directly via google-genai."""
        api_key, env_name = self._get_api_key("gemini", "GEMINI_API_KEY")
        if not api_key:
            return QueryResult(ok=False, error=f"Missing API key in env var {env_name}")

        try:
            from google import genai  # noqa: WPS433 (lazy import)
        except ImportError:
            return QueryResult(
                ok=False,
                error="google-genai SDK not installed; add package 'google-genai'",
            )

        backend_cfg = self._get_backend_config("gemini")
        model = backend_cfg.get("model", "gemini-2.5-pro")
        prompt = self._prompt_with_history(user_text, self._system_prompt)

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=model, contents=prompt)
            text = self._extract_text_content(getattr(response, "text", ""))
            if not text:
                candidates = getattr(response, "candidates", [])
                if candidates:
                    first_candidate = candidates[0]
                    content = getattr(first_candidate, "content", None)
                    if content is not None:
                        text = self._extract_text_content(getattr(content, "parts", []))
            if not text:
                return QueryResult(ok=False, error="gemini returned empty content")
            return QueryResult(ok=True, text=text)
        except Exception as exc:  # pragma: no cover - network/sdk surface
            log.error("[QueryRouter] gemini failed: %s", exc)
            return QueryResult(ok=False, error=f"gemini request failed: {exc}")
