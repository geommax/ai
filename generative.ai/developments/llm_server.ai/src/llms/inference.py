"""Inference engine — auto-routing frontend for multiple backends.

Supports:
  • **transformers** — safetensors / pytorch / bin models
  • **llama.cpp** — GGUF quantised models

The engine auto-detects the correct backend from the model path or
repo contents and delegates all calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.llms.backends.base import BaseBackend
from src.llms.backends.llms_transformers import TransformersBackend
from src.llms.backends.llms_llama_cpp import LlamaCppBackend

log = logging.getLogger("llm_daemon")


def _resolve_local_path(model_id: str) -> str:
    """Resolve an HF repo-id to its local cache snapshot path.

    If *model_id* already points to an existing file or directory it
    is returned unchanged.  Otherwise we look it up in the HF Hub
    cache (``scan_cache_dir``).
    """
    p = Path(model_id)
    if p.exists():
        return model_id

    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                # Pick the newest revision (most recently modified)
                best = max(repo.revisions, key=lambda r: r.last_modified)
                snapshot = str(best.snapshot_path)
                log.info(
                    "Resolved '%s' → %s", model_id, snapshot,
                )
                return snapshot
    except Exception:
        pass

    return model_id


def detect_backend(model_path: str) -> str:
    """Return ``'llama.cpp'`` or ``'transformers'`` for *model_path*.

    Detection rules (in priority order):
      1. Path ends in ``.gguf``  → llama.cpp
      2. Directory contains any ``.gguf`` file → llama.cpp
      3. Otherwise → transformers
    """
    p = Path(model_path)

    # Direct .gguf file
    if p.is_file() and p.suffix == ".gguf":
        return "llama.cpp"

    # Directory with .gguf inside
    if p.is_dir():
        if any(p.rglob("*.gguf")):
            return "llama.cpp"

    return "transformers"


class InferenceEngine:
    """Auto-routing inference engine.

    On ``load_model`` it detects whether the model is GGUF or
    safetensors/pytorch format and instantiates the correct backend.
    All downstream callers (daemon, FastAPI server) work through
    the same interface unchanged.
    """

    def __init__(self) -> None:
        self._backend: Optional[BaseBackend] = None
        self._active_backend_name: Optional[str] = None

    # Backend name constants
    BACKENDS = ("auto", "transformers", "llama.cpp")

    # ── Load / Unload ──────────────────────────────────────────────────
    def load_model(
        self,
        model_id: str,
        *,
        force_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Load *model_id* with automatic or forced backend selection.

        Parameters
        ----------
        model_id:
            HuggingFace repo-id **or** local path.
        force_backend:
            ``None`` / ``'auto'`` → auto-detect from model files.
            ``'transformers'``   → force the Transformers backend.
            ``'llama.cpp'``      → force the llama.cpp (GGUF) backend.
        **kwargs:
            Forwarded to the backend's ``load()``
            (e.g. ``n_gpu_layers``, ``n_ctx`` for llama.cpp).
        """
        self.unload_model()

        # Resolve HF repo-id → local snapshot path so both
        # detect_backend and the backend itself can inspect files.
        local_path = _resolve_local_path(model_id)

        if force_backend and force_backend != "auto":
            backend_name = force_backend
            log.info("Forced backend '%s' for %s", backend_name, model_id)
        else:
            backend_name = detect_backend(local_path)
            log.info("Detected backend '%s' for %s", backend_name, model_id)

        if backend_name == "llama.cpp":
            backend: BaseBackend = LlamaCppBackend()
        else:
            backend = TransformersBackend()

        backend.load(local_path, **kwargs)
        self._backend = backend
        self._active_backend_name = backend_name
        # Store the human-friendly model_id (repo-id) for display
        self._backend._model_id = model_id

    def reload_with_backend(self, backend_name: str, **kwargs: Any) -> None:
        """Switch the currently loaded model to a different backend.

        The model is unloaded and reloaded using *backend_name*.
        Raises ``RuntimeError`` if no model is currently loaded.
        """
        if not self.is_loaded or self._backend is None:
            raise RuntimeError("No model loaded — load a model first.")
        model_id = self._backend.model_id
        if model_id is None:
            raise RuntimeError("Current model has no model_id.")
        self.load_model(model_id, force_backend=backend_name, **kwargs)

    def unload_model(self) -> None:
        """Release current backend and reclaim resources."""
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        self._active_backend_name = None

    # ── Generation ─────────────────────────────────────────────────────
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text continuation for *prompt*."""
        if self._backend is None:
            raise RuntimeError("No model loaded — load a model first.")
        return self._backend.generate(prompt, **kwargs)

    def chat_generate(self, messages: list[dict], **kwargs: Any) -> str:
        """Generate a response from chat *messages*."""
        if self._backend is None:
            raise RuntimeError("No model loaded — load a model first.")
        return self._backend.chat_generate(messages, **kwargs)

    # ── Introspection ──────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._backend is not None and self._backend.is_loaded

    @property
    def model_id(self) -> str | None:
        return self._backend.model_id if self._backend else None

    @property
    def active_backend(self) -> str | None:
        """Name of the currently active backend (``'transformers'`` / ``'llama.cpp'``)."""
        return self._active_backend_name

    def device_info(self) -> Dict[str, str]:
        if self._backend is not None:
            info = self._backend.device_info()
            info["backend"] = self._backend.backend_name
            return info
        # Fallback — no model loaded
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return {
                    "type": "CUDA",
                    "name": props.name,
                    "memory": f"{props.total_memory / 1024**3:.1f} GB",
                    "backend": "none",
                }
        except ImportError:
            pass
        return {"type": "CPU", "name": "—", "memory": "—", "backend": "none"}
