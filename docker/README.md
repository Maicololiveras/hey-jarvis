# Jarvis Docker Infrastructure

Containerized backend for the Jarvis voice assistant.

## Architecture

```
HOST (Windows)                    DOCKER
+-----------------------+         +------------------------+
| Jarvis Frontend       |  gRPC   | jarvis-backend :50052  |
| - Mic / Wake word     | ------> | - STT (faster-whisper) |
| - VAD (silero)        |         | - QueryRouter          |
| - Audio playback      | <------ | - TTS (edge-tts/PyAV)  |
| - PyQt6 UI            |         +------------------------+
+-----------------------+                    |
                                             | gRPC
                                             v
                                  +------------------------+
                                  | maix-engine :50051     |
                                  | - llama-cpp-python     |
                                  | - GGUF model inference |
                                  +------------------------+
```

## Quick Start

### 1. Configure environment

```bash
cd C:\SDK\jarvis\docker
cp .env.example .env
# Edit .env with your API keys and model paths
```

### 2. Build and run

```bash
docker compose up -d --build
```

### 3. Verify services

```bash
docker compose ps
docker compose logs -f jarvis-backend
```

### 4. Connect frontend

From the host machine, run the Jarvis frontend pointing to `localhost:50052`:

```bash
cd C:\SDK\jarvis
python -m jarvis --backend-host localhost --backend-port 50052
```

## Connecting from Another PC

### Option A: Direct network (same LAN)

On the Docker host, ensure port 50052 is open in Windows Firewall:

```powershell
New-NetFirewallRule -DisplayName "Jarvis gRPC" -Direction Inbound -Port 50052 -Protocol TCP -Action Allow
```

From the remote PC:

```bash
python -m jarvis --backend-host <docker-host-ip> --backend-port 50052
```

### Option B: SSH tunnel

From the remote PC:

```bash
ssh -L 50052:localhost:50052 user@<docker-host-ip>
# Then connect frontend to localhost:50052
```

### Option C: Cloudflare Tunnel (internet access)

On the Docker host:

```bash
cloudflared tunnel --url tcp://localhost:50052
```

On the remote PC, use the tunnel URL provided by cloudflared.

## GPU Support (CUDA)

To build maix-engine with CUDA acceleration:

1. Use an NVIDIA GPU-capable Docker runtime (install nvidia-container-toolkit)
2. Set in `.env`:
   ```
   CMAKE_ARGS=-DGGML_CUDA=on
   N_GPU_LAYERS=35
   ```
3. Add to `docker-compose.yml` under `maix-engine`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```
4. Rebuild: `docker compose up -d --build maix-engine`

## Useful Commands

```bash
# Rebuild a single service
docker compose build jarvis-backend

# View logs
docker compose logs -f --tail=100 jarvis-backend

# Restart a service
docker compose restart jarvis-backend

# Stop everything
docker compose down

# Stop and remove volumes
docker compose down -v
```
