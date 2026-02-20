"""llama.cpp backend — GGUF quantised models via llama-cpp-python."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.llms.backends.base import BaseBackend

log = logging.getLogger("llm_daemon")


def _find_gguf_file(path: str) -> str:
    """Resolve a concrete ``.gguf`` file from *path*.

    *path* may be:
      • a direct ``.gguf`` file path
      • a directory containing one or more ``.gguf`` files
      • an HF cache snapshot directory (model repo)

    Returns the path to the best-match ``.gguf`` file.
    """
    p = Path(path)

    if p.is_file() and p.suffix == ".gguf":
        return str(p)

    if p.is_dir():
        # Collect all .gguf files recursively
        gguf_files = sorted(p.rglob("*.gguf"), key=lambda f: f.stat().st_size)
        if gguf_files:
            # Prefer the largest file (usually the full model weight)
            return str(gguf_files[-1])

    raise FileNotFoundError(
        f"No .gguf file found in '{path}'. "
        "Please download a GGUF model first."
    )


class LlamaCppBackend(BaseBackend):
    """Run GGUF models via ``llama-cpp-python``."""

    def __init__(self) -> None:
        self._llm: Any = None
        self._model_id: Optional[str] = None
        self._model_path: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────
    def load(self, model_path: str, **kwargs: Any) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            ) from exc

        self.unload()

        gguf_path = _find_gguf_file(model_path)
        log.info("Loading GGUF: %s", gguf_path)

        # Determine GPU layers
        n_gpu_layers = kwargs.pop("n_gpu_layers", -1)  # -1 = offload all
        n_ctx = kwargs.pop("n_ctx", 4096)

        self._llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
            **kwargs,
        )

        self._model_path = gguf_path
        # Use the parent directory name or file stem as model_id
        self._model_id = model_path
        log.info("GGUF model loaded: %s (ctx=%d, gpu_layers=%s)",
                 gguf_path, n_ctx, n_gpu_layers)

    def unload(self) -> None:
        if self._llm is not None:
            try:
                del self._llm
            except Exception:
                pass
            self._llm = None

        self._model_id = None
        self._model_path = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Generation ─────────────────────────────────────────────────
    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self._llm is None:
            raise RuntimeError("No model loaded — load a GGUF model first.")

        gen_kwargs: Dict[str, Any] = {
            "max_tokens": int(kwargs.get("max_tokens", 512)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "top_k": int(kwargs.get("top_k", 50)),
            "repeat_penalty": float(kwargs.get("repetition_penalty", 1.1)),
        }

        result = self._llm.create_completion(prompt, **gen_kwargs)
        return result["choices"][0]["text"]

    def chat_generate(self, messages: list[dict], **kwargs: Any) -> str:
        if self._llm is None:
            raise RuntimeError("No model loaded — load a GGUF model first.")

        gen_kwargs: Dict[str, Any] = {
            "max_tokens": int(kwargs.get("max_tokens", 512)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "top_k": int(kwargs.get("top_k", 50)),
            "repeat_penalty": float(kwargs.get("repetition_penalty", 1.1)),
        }

        # Format messages for llama.cpp
        chat_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]

        result = self._llm.create_chat_completion(
            messages=chat_messages, **gen_kwargs
        )
        return result["choices"][0]["message"]["content"]

    # ── Introspection ──────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "llama.cpp"

    def device_info(self) -> Dict[str, str]:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return {
                    "type": "CUDA",
                    "name": props.name,
                    "memory": f"{props.total_memory / 1024**3:.1f} GB",
                }
        except ImportError:
            pass
        return {"type": "CPU", "name": "—", "memory": "—"}
