# LLM Server.AI

Local LLM inference server with a terminal UI for management and an OpenAI-compatible REST API.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Framework](https://img.shields.io/badge/TUI-Textual-green)
![API](https://img.shields.io/badge/API-FastAPI-teal)

## Overview

A self-hosted LLM server that runs Hugging Face models on your local GPU. Manage everything — model downloads, API keys, generation parameters — through a Tokyo Night–themed terminal UI, and serve completions via an OpenAI-compatible API.

### Architecture

```
┌─────────────────┐         Unix Socket (JSON)         ┌─────────────────────┐
│   Textual TUI   │  ◄──────────────────────────────►  │   Daemon Process    │
│  (thin client)  │                                    │                     │
└─────────────────┘                                    │  ├─ InferenceEngine │
                                                       │  ├─ ModelManager    │
┌─────────────────┐         HTTP  (REST)               │  ├─ FastAPI Server  │
│   API Clients   │  ◄──────────────────────────────►  │  └─ SQLite DB       │
│  (curl, apps)   │                                    │                     │
└─────────────────┘                                    └─────────────────────┘
```

- **Daemon** — long-running background process that owns all heavy state (GPU model, API server, database)
- **TUI** — lightweight Textual frontend that talks to the daemon over a Unix socket
- **API** — OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints, secured with API keys

### TUI Screens

| Screen | Description |
|--------|-------------|
| Dashboard | Server status, loaded model, quick start/stop |
| Models | Search Hugging Face Hub, download, load/unload, delete models |
| API Keys | Generate, list, activate/revoke, delete API keys |
| Testing | Interactive prompt → response playground |
| Tuning | Adjust generation parameters (temperature, top_p, top_k, etc.) |
| Settings | HF login, server config, model directory, preferences |

## Setup

```bash
# Create conda environment
conda create -n ragllm python=3.10 -y
conda activate ragllm

# Install PyTorch (CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start (launches daemon + TUI)
python run.py

# Stop background daemon
python run.py --stop

# Check daemon status
python run.py --status

# Run daemon in foreground (for debugging)
python run.py --daemon
```

### TUI Keybindings

| Key | Action |
|-----|--------|
| `d` | Dashboard |
| `m` | Models |
| `k` | API Keys |
| `t` | Testing |
| `o` | Tuning |
| `s` | Settings |
| `q` | Quit TUI |

> Quitting the TUI does **not** stop the daemon. Use `python run.py --stop` to fully shut down.

### API

The server exposes OpenAI-compatible endpoints at `http://127.0.0.1:8000` (default).

```bash
# Chat completion
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer llm-YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Tuning parameters (`temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty`, `do_sample`) can be sent per-request to override server defaults.

## Data

All state is stored under `~/.config/llm_server_ai/`:

| File | Purpose |
|------|---------|
| `config.json` | Server configuration |
| `llm_server.db` | API keys (SQLite) |
| `daemon.pid` | Daemon PID file |
| `daemon.sock` | Unix domain socket |
| `daemon.log` | Daemon log output |

Model weights are cached in the Hugging Face hub cache (`~/.cache/huggingface/hub/` by default, configurable in Settings).
