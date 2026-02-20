# ═══════════════════════════════════════════════════════════════════════
#  LLM Server.AI — Docker Image
#  GPU-accelerated local LLM server with Textual TUI
# ═══════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System deps ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential cmake git curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# ── Install llama-cpp-python with CUDA (needs build tools) ─────────────
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV CUDACXX=/usr/local/cuda/bin/nvcc
RUN pip install --no-cache-dir llama-cpp-python==0.3.16

# ═══════════════════════════════════════════════════════════════════════
#  Runtime stage — smaller image
# ═══════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TERM=xterm-256color

# ── System deps (runtime only) ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# ── Copy llama-cpp-python from builder ─────────────────────────────────
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages

# ── Python dependencies ───────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ── App code ───────────────────────────────────────────────────────────
WORKDIR /app
COPY run.py .
COPY src/ src/

# ── Dirs for volume mounts ─────────────────────────────────────────────
RUN mkdir -p /root/.cache/huggingface/hub \
    && mkdir -p /root/.config/llm_server_ai

# ── Volumes (model cache + config/db) ─────────────────────────────────
VOLUME ["/root/.cache/huggingface/hub", "/root/.config/llm_server_ai"]

# ── API server port ────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# ── Entrypoint — launch TUI ───────────────────────────────────────────
CMD ["python", "run.py"]
