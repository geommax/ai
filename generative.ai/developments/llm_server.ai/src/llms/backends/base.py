"""Abstract base class that every inference backend must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseBackend(ABC):
    """Contract for a model-inference backend.

    Each backend is responsible for loading one model at a time,
    generating text from prompts or chat messages, and releasing
    resources on unload.
    """

    # ── Lifecycle ──────────────────────────────────────────────────
    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> None:
        """Load a model from *model_path* (local path or repo-id)."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and reclaim all resources (GPU memory etc.)."""

    # ── Generation ─────────────────────────────────────────────────
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return generated text for a plain-text *prompt*."""

    @abstractmethod
    def chat_generate(self, messages: list[dict], **kwargs: Any) -> str:
        """Return generated text from a list of chat *messages*.

        Each message dict has ``role`` and ``content`` keys.
        """

    # ── Introspection ──────────────────────────────────────────────
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """``True`` if a model is currently loaded and ready."""

    @property
    @abstractmethod
    def model_id(self) -> str | None:
        """Identifier of the currently loaded model, or ``None``."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend name (e.g. ``'transformers'``)."""

    @abstractmethod
    def device_info(self) -> Dict[str, str]:
        """Return dict with ``type``, ``name``, ``memory`` of the device."""
