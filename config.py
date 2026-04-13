"""Hey Jarvis configuration — read/write ~/.claudio/config.yaml.

Format (version 2):

    version: 2
    jarvis:
      wake_word: "hey jarvis"
      wake_word_model: "hey_jarvis_v1"
      silence_timeout_seconds: 5
      language: "auto"

    audio:
      sample_rate: 16000
      channels: 1
      chunk_duration_ms: 80
      device_override: null
      pre_gain: 5.0
      highpass_cutoff_hz: 80
      agc_target_peak: 0.9
      agc_max_gain: 20.0
      agc_min_peak: 0.005

    stt:
      engine: "faster-whisper"
      model_path: "D:/Transcripcion con ia/whisper_models/medium"
      fast_model_path: "D:/Transcripcion con ia/whisper_models/base"
      model: "whisper-large-v3"
      api_key_env: "GROQ_API_KEY"
      use_precise_pass: false
      device: "cpu"
      compute_type: "int8"

    tts:
      engine: "edge-tts"
      voice_es: "es-CO-GonzaloNeural"
      voice_en: "en-US-GuyNeural"
      rate: "+5%"
      playback: "ffplay"
      offline_fallback: true

    query:
      default_backend: "claude-api"
      timeout_seconds: 60
      backends:
        claude-p:
          command: "claude"
          args: ["-p"]
        claude-api:
          model: "claude-sonnet-4-6"
          api_key_env: "ANTHROPIC_API_KEY"
        openai:
          model: "gpt-4o"
          api_key_env: "OPENAI_API_KEY"
        gemini:
          model: "gemini-2.5-pro"
          api_key_env: "GEMINI_API_KEY"
        local-qwen:
          host: "http://localhost"
          port: 8081
          model_path: "..."

    ui:
      enabled: true
      shape: "circle"
      diameter: 200
      position: "center"
      colors:
        primary: "#00BFFF"
        glow: "#FFFFFF"
        background: "rgba(0, 20, 40, 0.6)"
      fps: 60

    engram:
      enabled: true
      auto_save_sessions: true
      context_on_query: true

    wake_word:
      engine: "openwakeword"
      model: "hey_jarvis"
      threshold: 0.04
      consecutive_frames: 2
      extra_gain: 2.0
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path.home() / ".claudio"
CONFIG_FILE = "config.yaml"
CONFIG_PATH = CONFIG_DIR / CONFIG_FILE
EXAMPLE_CONFIG_PATH = Path(__file__).with_name("config.yaml.example")


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 2,
    "jarvis": {
        "wake_word": "hey jarvis",
        "wake_word_model": "hey_jarvis_v1",
        "silence_timeout_seconds": 5,
        "language": "auto",
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_duration_ms": 80,
        "device_override": None,
        "pre_gain": 5.0,
        "highpass_cutoff_hz": 80,
        "agc_target_peak": 0.9,
        "agc_max_gain": 20.0,
        "agc_min_peak": 0.005,
    },
    "stt": {
        "engine": "faster-whisper",
        "model_path": "D:/Transcripcion con ia/whisper_models/medium",
        "fast_model_path": "D:/Transcripcion con ia/whisper_models/base",
        "model": "whisper-large-v3",
        "api_key_env": "GROQ_API_KEY",
        "use_precise_pass": False,
        "device": "cpu",
        "compute_type": "int8",
    },
    "tts": {
        "engine": "edge-tts",
        "voice_es": "es-CO-GonzaloNeural",
        "voice_en": "en-US-GuyNeural",
        "rate": "+5%",
        "playback": "ffplay",
        "offline_fallback": True,
    },
    "query": {
        "default_backend": "claude-api",
        "timeout_seconds": 60,
        "backends": {
            "claude-p": {
                "command": "claude",
                "args": ["-p"],
            },
            "claude-api": {
                "model": "claude-sonnet-4-6",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
            "openai": {
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
            },
            "gemini": {
                "model": "gemini-2.5-pro",
                "api_key_env": "GEMINI_API_KEY",
            },
            "local-qwen": {
                "host": "http://localhost",
                "port": 8081,
                "model_path": "D:\\models\\Qwen\\Qwen2.5-Coder-14B-Instruct-GGUF\\qwen2.5-coder-14b-instruct-q4_0.gguf",
            },
        },
    },
    "ui": {
        "enabled": True,
        "shape": "circle",
        "diameter": 200,
        "position": "center",
        "colors": {
            "primary": "#00BFFF",
            "glow": "#FFFFFF",
            "background": "rgba(0, 20, 40, 0.6)",
        },
        "fps": 60,
    },
    "engram": {
        "enabled": True,
        "auto_save_sessions": True,
        "context_on_query": True,
    },
    "wake_word": {
        "engine": "openwakeword",
        "model": "hey_jarvis",
        "threshold": 0.04,
        "consecutive_frames": 2,
        "extra_gain": 2.0,
    },
}


# ---------------------------------------------------------------------------
# Core load / save / merge
# ---------------------------------------------------------------------------


def load() -> dict[str, Any]:
    """Load config from ~/.claudio/config.yaml, or return defaults."""
    if not CONFIG_PATH.exists():
        if EXAMPLE_CONFIG_PATH.exists():
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EXAMPLE_CONFIG_PATH, CONFIG_PATH)
        else:
            return DEFAULT_CONFIG.copy()

    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    # Merge with defaults so missing keys fall back to sane values.
    merged = DEFAULT_CONFIG.copy()
    _deep_update(merged, loaded)
    return merged


def save(config: dict[str, Any]) -> None:
    """Write the config to ~/.claudio/config.yaml, creating the folder."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)


def _deep_update(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Typed accessor helpers
# ---------------------------------------------------------------------------


def get_wake_word_config() -> dict[str, Any]:
    """Return the 'wake_word' section of the config."""
    return load().get("wake_word", {})


def get_stt_config() -> dict[str, Any]:
    """Return the 'stt' section of the config."""
    return load().get("stt", {})


def get_tts_config() -> dict[str, Any]:
    """Return the 'tts' section of the config."""
    return load().get("tts", {})


def get_query_config() -> dict[str, Any]:
    """Return the 'query' section of the config."""
    return load().get("query", {})


def get_ui_config() -> dict[str, Any]:
    """Return the 'ui' section of the config."""
    return load().get("ui", {})


def get_audio_config() -> dict[str, Any]:
    """Return the 'audio' section of the config."""
    return load().get("audio", {})


def get_engram_config() -> dict[str, Any]:
    """Return the 'engram' section of the config."""
    return load().get("engram", {})


# ---------------------------------------------------------------------------
# Status printer
# ---------------------------------------------------------------------------


def print_status() -> int:
    """Pretty-print the current config — for `jarvis status`."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    cfg = load()

    console.print(
        "[bold cyan]Hey Jarvis — current config (v{ver})[/bold cyan]".format(
            ver=cfg.get("version", "?"),
        )
    )
    console.print(f"  Config file: [dim]{CONFIG_PATH}[/dim]")

    # Jarvis core
    jarvis = cfg.get("jarvis", {})
    console.print(
        f"  Wake word: [yellow]{jarvis.get('wake_word')}[/yellow] "
        f"(model={jarvis.get('wake_word_model')}, "
        f"silence_timeout={jarvis.get('silence_timeout_seconds')}s, "
        f"language={jarvis.get('language')})"
    )

    # Wake word engine
    ww = cfg.get("wake_word", {})
    console.print(
        f"  Wake word engine: [yellow]{ww.get('engine')}/{ww.get('model')}[/yellow] "
        f"(threshold={ww.get('threshold')}, "
        f"frames={ww.get('consecutive_frames')}, "
        f"extra_gain={ww.get('extra_gain')})"
    )

    # Audio
    audio = cfg.get("audio", {})
    console.print(
        f"  Audio: [yellow]{audio.get('sample_rate')}Hz[/yellow] "
        f"ch={audio.get('channels')} "
        f"chunk={audio.get('chunk_duration_ms')}ms "
        f"pre_gain={audio.get('pre_gain')} "
        f"device={audio.get('device_override') or '[dim]auto[/dim]'}"
    )

    # STT
    stt = cfg.get("stt", {})
    console.print(
        f"  STT: [yellow]{stt.get('engine')}[/yellow] "
        f"({stt.get('device')}/{stt.get('compute_type')}) "
        f"-> {stt.get('model_path') or '[red]NOT SET[/red]'}"
    )

    # TTS
    tts = cfg.get("tts", {})
    console.print(
        f"  TTS: [yellow]{tts.get('engine')}[/yellow] "
        f"es={tts.get('voice_es')} en={tts.get('voice_en')} "
        f"rate={tts.get('rate')} playback={tts.get('playback')} "
        f"offline_fallback={tts.get('offline_fallback')}"
    )

    # Query backends
    query = cfg.get("query", {})
    backends = query.get("backends", {})
    console.print(
        f"  Query: default=[yellow]{query.get('default_backend')}[/yellow] "
        f"timeout={query.get('timeout_seconds')}s"
    )
    if backends:
        table = Table(title="Query backends", show_lines=False)
        table.add_column("Backend", style="cyan")
        table.add_column("Details", style="dim")
        for name, bcfg in backends.items():
            if "command" in bcfg:
                details = f"{bcfg['command']} {' '.join(bcfg.get('args', []))}"
            elif "host" in bcfg:
                details = f"{bcfg['host']}:{bcfg.get('port', '?')}"
            else:
                details = str(bcfg)
            table.add_row(name, details)
        console.print(table)

    # UI
    ui = cfg.get("ui", {})
    console.print(
        f"  UI: enabled={ui.get('enabled')} "
        f"{ui.get('shape')} {ui.get('diameter')}px "
        f"pos={ui.get('position')} fps={ui.get('fps')}"
    )

    # Engram
    engram = cfg.get("engram", {})
    console.print(
        f"  Engram: enabled={engram.get('enabled')} "
        f"auto_save={engram.get('auto_save_sessions')} "
        f"context_on_query={engram.get('context_on_query')}"
    )

    return 0
