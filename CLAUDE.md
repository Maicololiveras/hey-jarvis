# Jarvis — AI Voice Assistant

## Multi-Terminal Coordination Protocol (MANDATORY)

This repo is worked on by multiple AI agents simultaneously (OpenCode terminals orchestrated by Claude Code). To prevent agents from overwriting each other's work, this protocol is MANDATORY.

### Terminal Roles

| Terminal | Role | Responsibilities |
|----------|------|-----------------|
| **T1 (Claude Code)** | Orchestrator | Plans, verifies, validates, coordinates. Merges branches. |
| **T2 (OpenCode)** | Validator/Implementor | Reviews, validates, or implements in separate branches. |
| **T3 (OpenCode)** | Implementor | Writes code, fixes bugs, implements features in separate branches. |

### Flow

```
T1 assigns task → T2/T3 create feature branch → implement → commit+push → T1 validates → T1 merges to master or rejects
```

### Branch Strategy

- **master**: production — only merged after validation
- **feature/***: one per task, created from master, merged by T1 after review
- T2 and T3 work on SEPARATE branches simultaneously — never same branch
- Use keyboard shortcuts (Ctrl+Alt+1/2/3) to switch tabs, NOT mouse clicks

### Before Writing Code (MANDATORY)

1. `git checkout master && git pull origin master` before creating feature branch
2. Search engram: `mem_search(query: "jarvis terminal changes", project: "jarvis")`
3. `git log --oneline -5` to see latest commits
4. Do NOT rewrite files another terminal already changed without reading the diff first

### After Completing Work (MANDATORY)

1. `py_compile` all changed files
2. Commit and push the feature branch
3. Use `Ctrl+Alt+1` to go to T1 tab and type feedback summary
4. Save to engram with `topic_key: jarvis/terminal-{N}-changes`, `project: jarvis`

## Architecture

### Audio Pipeline (dual-path)
```
Mic → raw chunk (80ms @ 16kHz)
  ├─ Path 1: raw → highpass+AGC → openwakeword (wake word, NO noise filter)
  └─ Path 2: raw → highpass+AGC → noisereduce → Silero VAD → SegmentEvent → STT
```

### Query Pipeline (async two-stage)
```
SegmentEvent → [Query Queue] → Worker Thread (STT→EchoDetect→QueryRouter) → [Response Queue] → TTS
```

### Components
- **State machine**: DORMIDO / ACTIVO with session_id (uuid4), `state_machine.py`
- **Wake word**: openwakeword `hey_jarvis`, threshold 0.08, consecutive_frames 1
- **STT**: faster-whisper, prefers CUDA (RTX 4060), fallback CPU. Forced Spanish.
- **Echo cancellation**: text-based fuzzy matching (rapidfuzz), `echo_detector.py`
- **Noise reduction**: noisereduce (DeepFilterNet fallback planned), dual-path
- **Query backends**: claude-p → opencode → groq → maix-engine (gRPC)
- **TTS**: edge-tts + sounddevice (internal playback), ffplay fallback, pyttsx3 offline fallback
- **UI**: PyQt6 frameless transparent overlay, waveform animation, backend selector (double-click)
- **Session**: uuid4 per wake session, idempotency for duplicate wakes

### Fallback Chains
- **Query**: claude-p → opencode → groq → maix-engine
- **TTS playback**: sounddevice → ffplay → pyttsx3
- **Noise reduction**: DeepFilterNet → noisereduce → raw audio
- **STT device**: CUDA → CPU

## Config

Runtime config at `~/.claudio/config.yaml`. Defaults in `config.py`.

Key tuning values:
- `wake_word.threshold`: 0.08
- `wake_word.consecutive_frames`: 1
- `wake_word.extra_gain`: 2.0
- `audio.pre_gain`: 3.0
- `audio.agc_max_gain`: 20.0
- `audio.noise_reduce`: true
- `stt.device`: cuda (fallback cpu)
- `stt.compute_type`: int8_float16

## Voice Commands
- **"Jarvis dormido"** (or "duerme", "para", "sleep"): finish queue then sleep

## Development Rules

- Never build after changes
- Always commit and push after completing work
- Use `python -m py_compile <file>` to verify syntax without running
- Launch with: `cd C:\SDK && python -u -m jarvis`
- Logs at: `C:\SDK\jarvis\jarvis.log`
- Feature branches: `git checkout -b feature/<name>` from master
- Merge only after T1 validation
