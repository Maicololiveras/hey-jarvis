"""Engram memory bridge — context enrichment and session persistence."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class EngramBridge:
    """Enriches queries with Engram memory context and saves session summaries."""

    def __init__(self, config: dict, query_config: dict) -> None:
        self._enabled = config.get("enabled", True)
        self._auto_save = config.get("auto_save_sessions", True)
        self._context_on_query = config.get("context_on_query", True)
        self._timeout = query_config.get("timeout_seconds", 60)
        self._claude_cmd = query_config.get("backends", {}).get("claude-p", {}).get("command", "claude")
        log.info("[EngramBridge] enabled=%s, auto_save=%s, context_on_query=%s",
                 self._enabled, self._auto_save, self._context_on_query)

    def enrich_prompt(self, user_query: str, language: str = "es") -> str:
        """Wrap user query with Engram context-search instructions.

        Returns the enriched prompt string. If engram is disabled,
        returns the original query unchanged.
        """
        lang_name = "Spanish" if language == "es" else f"the detected language ({language})"
        lang_line = f"IMPORTANT: You MUST respond in {lang_name}.\n\n"

        if not self._enabled or not self._context_on_query:
            return f"{lang_line}User question: {user_query}"

        # Extract keywords for search (simple: take words > 3 chars)
        keywords = " ".join(w for w in user_query.split() if len(w) > 3)[:100]

        return (
            f"{lang_line}"
            "BEFORE answering, search Engram for relevant context:\n"
            "1. Call mem_context(limit: 10) for recent session history\n"
            f'2. Call mem_search(query: "{keywords}") for related past work\n'
            "3. If you find relevant context, incorporate it naturally into your answer\n"
            "4. Do NOT mention the memory search to the user\n\n"
            "---\n\n"
            f"User question: {user_query}"
        )

    def save_session_summary(self, exchanges: list[dict]) -> None:
        """Save a session summary to Engram via claude -p.

        Args:
            exchanges: List of dicts with keys 'user', 'response', 'language', 'timestamp'
        """
        if not self._enabled or not self._auto_save:
            log.debug("[EngramBridge] Session save skipped (disabled)")
            return

        if not exchanges:
            log.debug("[EngramBridge] No exchanges to save")
            return

        # Build summary
        summary_lines = ["## Jarvis Voice Session Summary\n"]
        for i, ex in enumerate(exchanges, 1):
            user_text = ex.get("user", "")[:200]
            resp_text = ex.get("response", "")[:200]
            lang = ex.get("language", "unknown")
            summary_lines.append(f"### Exchange {i} (lang: {lang})")
            summary_lines.append(f"**User**: {user_text}")
            summary_lines.append(f"**Jarvis**: {resp_text}\n")

        summary = "\n".join(summary_lines)

        prompt = (
            "Save this voice session summary to Engram memory. "
            "Use mem_save with:\n"
            '- title: "Jarvis voice session summary"\n'
            '- type: "session_summary"\n'
            '- topic_key: "jarvis/voice-session"\n'
            '- project: "claudio-daemon"\n'
            f"- content:\n\n{summary}\n\n"
            "Just save it and respond with 'saved'."
        )

        try:
            subprocess.run(
                [self._claude_cmd, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
            log.info("[EngramBridge] Session summary saved (%d exchanges)", len(exchanges))
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.error("[EngramBridge] Failed to save session summary: %s", e)

    def is_enabled(self) -> bool:
        return self._enabled
