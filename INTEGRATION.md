# INTEGRATION — hey-jarvis (hey-jarvis-temp local)

> Este repo es la **Arma del wake word** del ecosistema ARISE. Es la
> Sombra `sombra-wake` del Mariscal Voz — detecta el comando "Hey
> ARISE" en el micrófono local y dispara el pipeline de voz.

**Documento maestro del ecosistema**:
[jarvis-core/docs/ecosystem/](https://github.com/Maicololiveras/jarvis-core/tree/main/docs/ecosystem)
**Documento del Mariscal Voz**:
`jarvis-core/docs/ecosystem/04-ARMY-STRUCTURE.md` §4.6
**Documento del routing**:
`jarvis-core/docs/ecosystem/05-MODEL-ROUTING.md`

---

## 1. Identidad rápida

| Campo | Valor |
|---|---|
| **Repo** | `hey-jarvis` (clon local `hey-jarvis-temp`) |
| **Owner / remote** | `Maicololiveras/hey-jarvis` |
| **Categoría ARISE** | **MCP / Arma** (con un poco de Sombra) |
| **Mariscal padre** | **M-Voz** |
| **Rango en la jerarquía 00B** | Sombra (`sombra-wake`) + Arma (wake word detector) |
| **Stack principal** | Python + (PyAudio / sounddevice) + porcupine / openWakeWord |
| **Estado** | scaffold — wake word funciona, falta consolidar con el pipeline ARISE |
| **Última auditoría ARISE** | 2026-06-15 |

---

## 2. Qué hace este repo dentro de ARISE

Daemon que corre **siempre** en background en la máquina del usuario
(Windows local de Maicol, no en el VPS). Escucha el micrófono 24/7,
detecta el wake word **"Hey ARISE"** (o "Hey Jarvis" como legacy), y
cuando lo detecta dispara el pipeline:

```
1. Mic local 24/7
2. Detecta "Hey ARISE" (~50 ms)
3. Notifica al Mariscal Voz vía evento (HTTP / IPC / engram event)
4. Mariscal Voz arma sombra-realtime-local (default)
   o sombra-realtime-cloud (si Maicol pidió visión)
5. Sombra graba con VAD, transcribe, invoca motor, responde
```

### Sombras que implementa

| Sombra | Cuándo se activa |
|---|---|
| `sombra-wake` | Continuamente — el daemon es la Sombra "always-on" |

### Armas que expone

| Arma | Tipo | Contrato I/O |
|---|---|---|
| `wake_event` | Evento (HTTP webhook / engram message) | — → `WakeEvent{timestamp, confidence, audio_clip_path}` |
| `set_wake_word` | CLI / config | `wake_word: "Hey ARISE"` o `"Hey Jarvis"` |
| `pause` / `resume` | CLI | Para silenciar el daemon temporalmente |

---

## 3. Cómo el Mariscal Voz lo invoca

```
[Daemon hey-jarvis corriendo en máquina local Maicol]
    ↓ (mic 24/7)
[Detecta "Hey ARISE"]
    ↓
[Emite WakeEvent → Mariscal Voz]
    ↓
[Mariscal Voz decide qué Sombra activar]
    │
    ├─ default → sombra-realtime-local (Ollama VPS)
    │
    └─ si user pidió visión → sombra-realtime-cloud (Gemini Live)
```

---

## 4. Conexión con el ecosistema

```yaml
# Este repo NO usa el motor router directamente — solo wake word
# detection es local y barato. El motor lo invocan los Sombras DESPUÉS.

corre_donde:
  - máquina del usuario (Windows / macOS / Linux) — siempre LOCAL
  - NO corre en el VPS (mic remoto = no sirve)

instalacion:
  - vía pip o binario standalone

eventos_emitidos:
  - wake_event → consumido por sombra-realtime-* del M-Voz
```

---

## 5. Memoria y persistencia

| Qué se guarda | Dónde | Scope |
|---|---|---|
| Histórico de wakes (timestamp + confidence) | SQLite local | `~/.hey-jarvis/log.db` |
| Config (wake word, mic device, sensitivity) | YAML | `~/.hey-jarvis/config.yaml` |
| Audio clips de wakes (debug) | filesystem | `~/.hey-jarvis/clips/` (rotar) |

---

## 6. Capacidad acotada (como Sombra del Mariscal Voz)

```jsonc
{
  "tools_permitidas": ["mic_local", "audio_clip_storage", "wake_event_emit"],
  "tools_prohibidas": ["llm_call", "whatsapp", "telegram", "ssh"],
  "max_sombras_hijas": 0,    // no delega
  "presupuesto_tokens": 0,    // no consume LLM
  "alcance_store": "~/.hey-jarvis/**"
}
```

---

## 7. Setup local

```bash
git clone https://github.com/Maicololiveras/hey-jarvis.git
cd hey-jarvis

pip install -e .

# Config
cp config.example.yaml ~/.hey-jarvis/config.yaml
# Editar:
#   wake_word: "Hey ARISE"
#   sensitivity: 0.5
#   mic_device: default
#   webhook_url: http://localhost:50101/wake-event   # → Mariscal Voz

# Arrancar como servicio
hey-jarvis daemon start

# O para test interactivo
hey-jarvis listen
```

### En el VPS NO se instala — corre solo en máquina del usuario.

---

## 8. SDD en este repo

Persistencia SDD: **engram**.

Topic keys:
- `sdd-init/hey-jarvis`
- `sdd/hey-jarvis/{change}/state`

Próximos SDD recomendados:

```bash
/sdd-new wake-word-arise          # cambiar default "Hey Jarvis" → "Hey ARISE"
/sdd-new webhook-mariscal-voz     # contrato de WakeEvent → M-Voz
/sdd-new multi-platform-build     # binarios para Win/Mac/Linux
```

---

## 9. Roadmap propio

- [ ] **Wake word "Hey ARISE"** como default (mantener "Hey Jarvis" como alias)
- [ ] **Webhook al Mariscal Voz** con payload estándar
- [ ] **Sensitivity adaptativa** según ruido ambiente
- [ ] **Multi-wake-word** — "Hey ARISE" y "OK ARISE" simultáneo
- [ ] **Binario standalone** para Windows / macOS / Linux
- [ ] **Service installer** (Windows service, launchd, systemd)
- [ ] **Privacy mode** — toggle físico de mute, indicador visual

---

## 10. Contratos críticos

- **El daemon NUNCA envía audio fuera de la máquina local** — solo emite
  metadatos del evento (timestamp + confidence). El audio se procesa
  después por `sombra-realtime-local` o se descarta.
- **Wake word es case-insensitive** pero solo en idiomas configurados.
  Por default ES y EN.
- **El evento `WakeEvent` debe llegar al Mariscal Voz en <200 ms** desde
  la detección, sino el user nota latencia.

---

## 11. Versionado de este INTEGRATION.md

| Versión | Fecha | Cambio |
|---|---|---|
| v0.1 | 2026-06-15 | Manifest inicial. Wake word ARISE como default propuesto. |
