# Jarvis — AI Voice Assistant

## Multi-Terminal Coordination Protocol (MANDATORY)

This repo is worked on by multiple AI agents simultaneously (OpenCode terminals orchestrated by Claude Code). To prevent agents from overwriting each other's work, this protocol is MANDATORY.

### Terminal Roles

| Terminal | Role | Responsibilities |
|----------|------|-----------------|
| **T1 (Claude Code)** | Orchestrator | Plans, verifies, validates, coordinates. Does NOT write code. |
| **T2 (OpenCode)** | Validator/Reviewer | Reviews what T3 implemented, finds bugs, gives feedback to T3. |
| **T3 (OpenCode)** | Implementor | Writes code, fixes bugs, implements features. |

### Flow

```
T1 assigns task to T3 → T3 implements → T3 saves to engram → T1 sends T2 to review → T2 validates and reports bugs → T1 sends feedback to T3 → T3 fixes → cycle repeats
```

### Before Writing Code (MANDATORY for T2 and T3)

1. **Search engram first**: `mem_search(query: "jarvis terminal changes", project: "jarvis")` to see what other terminals did recently.
2. **Check git log**: `git log --oneline -5` to see the latest commits and who made them.
3. **Do NOT rewrite files another terminal already changed** without reading the diff first.
4. **If another terminal is working on a task, do NOT start the same task.** Wait for it to finish.

### After Completing Work (MANDATORY for T2 and T3)

1. **Save to engram**: `mem_save` with:
   - `topic_key`: `jarvis/terminal-{N}-changes`
   - `project`: `jarvis`
   - Content: files touched, commits made, what changed, current state
2. **Commit and push** so other terminals see the changes.

### Rules

- One terminal owns one task at a time. No parallel work on the same file.
- T3 implements. T2 validates. T1 orchestrates. Roles do not overlap.
- If T2 finds a bug, it reports to T1 (orchestrator), who then instructs T3 to fix it.
- T2 NEVER fixes code directly — it only validates and reports.
- All coordination goes through engram and git history.

## Architecture

- **State machine**: DORMIDO (sleeping) / ACTIVO (active), defined in `state_machine.py`
- **Wake word**: openwakeword with `hey_jarvis` model
- **STT**: faster-whisper (base model for speed, medium for precision)
- **Query backends**: claude -p (default), claude-api, opencode, local-qwen — routed by `query_router.py`
- **TTS**: edge-tts + ffplay for playback, pyttsx3 as offline fallback
- **UI**: PyQt6 frameless transparent overlay with waveform

## Config

Runtime config at `~/.claudio/config.yaml`. Defaults in `config.py`.

Key tuning values:
- `wake_word.threshold`: 0.12 (lower = more sensitive, higher = fewer false positives)
- `wake_word.consecutive_frames`: 2 (frames above threshold needed to trigger)
- `audio.pre_gain`: 3.0 (microphone amplification multiplier)

## Development Rules

- Never build after changes
- Always commit and push after completing work
- Use `python -m py_compile <file>` to verify syntax without running
- Launch with: `cd C:\SDK && python -u -m jarvis`
- Logs at: `C:\SDK\jarvis\jarvis.log`
