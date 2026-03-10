"""
Model Manager — scan HuggingFace cache, load/unload SafeTensors & GGUF models.

Lazy-loading design: model is only loaded when the user clicks the Load Model button in the UI.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .config import ModelFormat


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class CachedModel:
    """Metadata for a single model found in the HuggingFace cache."""
    model_id: str                       # e.g. "Qwen/Qwen2.5-3B-Instruct"
    model_format: ModelFormat           # SAFETENSORS or GGUF
    snapshot_path: Path                 # absolute path to snapshot dir
    gguf_files: list[str] = field(default_factory=list)  # GGUF filenames

    @property
    def display_name(self) -> str:
        tag = "🟢 SafeTensors" if self.model_format == ModelFormat.SAFETENSORS else "🟠 GGUF"
        return f"[{tag}]  {self.model_id}"


@dataclass
class LoadedModelState:
    """Runtime state of the currently loaded model."""
    model_id: str
    model_format: ModelFormat
    model: object = None            # AutoModelForCausalLM  or  Llama
    tokenizer: object = None        # AutoTokenizer          (SafeTensors only)
    pipe: object = None             # transformers pipeline  (SafeTensors only)
    gguf_filename: str = ""         # loaded GGUF filename


# ── Cache Scanner ────────────────────────────────────────────────────────

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _model_id_from_dir(dirname: str) -> str:
    """'models--Qwen--Qwen2.5-3B-Instruct'  →  'Qwen/Qwen2.5-3B-Instruct'"""
    parts = dirname.replace("models--", "", 1).split("--")
    return "/".join(parts)


def _latest_snapshot(model_dir: Path) -> Optional[Path]:
    """Return the latest (newest modified) snapshot directory."""
    snap_root = model_dir / "snapshots"
    if not snap_root.is_dir():
        return None
    snapshots = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def _detect_format(snapshot: Path) -> tuple[ModelFormat, list[str]]:
    """
    Detect model format by inspecting file extensions in the snapshot directory.
    Returns (format, list_of_gguf_filenames).
    """
    gguf_files: list[str] = []
    has_safetensors = False

    for f in snapshot.iterdir():
        name = f.name.lower()
        if name.endswith(".gguf"):
            gguf_files.append(f.name)
        elif name.endswith(".safetensors"):
            has_safetensors = True

    if gguf_files:
        return ModelFormat.GGUF, sorted(gguf_files)
    if has_safetensors:
        return ModelFormat.SAFETENSORS, []

    # Fallback: check for bin weights (older format, treat as SafeTensors pipeline)
    for f in snapshot.iterdir():
        if f.name.endswith(".bin") and "pytorch" not in f.name.lower():
            return ModelFormat.SAFETENSORS, []

    return ModelFormat.SAFETENSORS, []


def scan_cached_models(cache_dir: Path | None = None) -> list[CachedModel]:
    """
    Scan ~/.cache/huggingface/hub/ for previously downloaded models
    and return a list of CachedModel instances.
    """
    root = cache_dir or HF_CACHE_DIR
    if not root.is_dir():
        return []

    results: list[CachedModel] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue

        snapshot = _latest_snapshot(entry)
        if snapshot is None:
            continue

        model_id = _model_id_from_dir(entry.name)
        fmt, gguf_files = _detect_format(snapshot)

        results.append(
            CachedModel(
                model_id=model_id,
                model_format=fmt,
                snapshot_path=snapshot,
                gguf_files=gguf_files,
            )
        )

    return results


# ── Model Loader / Unloader ─────────────────────────────────────────────

_state: Optional[LoadedModelState] = None


def get_state() -> Optional[LoadedModelState]:
    """Return current loaded model state (or None)."""
    return _state


def is_loaded() -> bool:
    return _state is not None and _state.model is not None


def _get_gpu_memory_mb() -> tuple[float, float] | None:
    """Return (used_MB, total_MB) for the current CUDA device, or None."""
    if not torch.cuda.is_available():
        return None
    used = torch.cuda.memory_allocated() / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    return used, total


def unload_model() -> str:
    """
    Thoroughly unload the current model and release all GPU/CPU memory.

    Steps:
      1. Record GPU memory before cleanup.
      2. For SafeTensors: remove accelerate dispatch hooks, move to CPU, delete.
         For GGUF: call .close() on the llama-cpp Llama object, then delete.
      3. Delete all references from the state object.
      4. Run multiple gc.collect() rounds to break circular references.
      5. Synchronize CUDA, empty cache, and run ipc_collect.
      6. Report actual memory freed.
    """
    global _state

    if _state is None:
        return "ℹ️ No model is loaded."

    old_id = _state.model_id
    old_format = _state.model_format
    gpu_before = _get_gpu_memory_mb()

    # ── 1. Format-specific deep cleanup ──────────────────────────────
    if old_format == ModelFormat.SAFETENSORS:
        _cleanup_safetensors_model(_state)
    else:
        _cleanup_gguf_model(_state)

    # ── 2. Clear all remaining references on the state object ────────
    _state.pipe = None
    _state.model = None
    _state.tokenizer = None
    _state = None

    # ── 3. Aggressive garbage collection (multiple rounds) ───────────
    for _ in range(3):
        gc.collect()

    # ── 4. CUDA cleanup ─────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    # Final GC round after CUDA cleanup
    gc.collect()

    # ── 5. Build status report ───────────────────────────────────────
    gpu_after = _get_gpu_memory_mb()
    report = f"✅ Model **{old_id}** unloaded."

    if gpu_before and gpu_after:
        freed = gpu_before[0] - gpu_after[0]
        report += (
            f"\n\n**GPU Memory:**  "
            f"{gpu_before[0]:.0f} MB → {gpu_after[0]:.0f} MB  "
            f"(freed **{freed:.0f} MB**)"
        )

    return report


def _cleanup_safetensors_model(state: LoadedModelState) -> None:
    """Deep cleanup for a transformers (SafeTensors) model."""
    model = state.model
    if model is None:
        return

    # Remove accelerate dispatch hooks (they hold tensor references)
    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(model)
    except Exception:
        pass

    # Untie any shared weight references
    try:
        if hasattr(model, "tie_weights"):
            # Some models tie lm_head ↔ embedding; untie to avoid dangling refs
            pass
    except Exception:
        pass

    # Move all parameters to CPU first to release GPU allocations
    try:
        model.to("cpu")
    except Exception:
        pass

    # Delete sub-components that may hold separate references
    for attr in ("lm_head", "model", "transformer", "encoder", "decoder"):
        try:
            if hasattr(model, attr):
                delattr(model, attr)
        except Exception:
            pass

    # Explicitly delete the model object
    del model
    state.model = None

    # Tokenizer can hold memory via fast-tokenizer rust backend
    if state.tokenizer is not None:
        del state.tokenizer
        state.tokenizer = None

    if state.pipe is not None:
        del state.pipe
        state.pipe = None


def _cleanup_gguf_model(state: LoadedModelState) -> None:
    """Deep cleanup for a llama-cpp-python (GGUF) model."""
    model = state.model
    if model is None:
        return

    # llama-cpp-python Llama objects have a close() or __del__ for C++ cleanup
    try:
        if hasattr(model, "close"):
            model.close()
    except Exception:
        pass

    # Some versions use _model attribute for the underlying C object
    try:
        if hasattr(model, "_model") and model._model is not None:
            del model._model
    except Exception:
        pass

    del model
    state.model = None


def load_model(
    model_id: str,
    model_format: ModelFormat,
    gguf_filename: str = "",
) -> str:
    """
    Load the model — either SafeTensors (Transformers) or GGUF (llama.cpp).

    Returns:
        status message string.
    """
    global _state

    # Unload any existing model first
    if is_loaded():
        unload_model()

    try:
        if model_format == ModelFormat.SAFETENSORS:
            return _load_safetensors(model_id)
        else:
            return _load_gguf(model_id, gguf_filename)
    except Exception as exc:
        _state = None
        return f"❌ Load failed: {exc}"


# ── Private loaders ─────────────────────────────────────────────────────

def _load_safetensors(model_id: str) -> str:
    global _state
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )

    _state = LoadedModelState(
        model_id=model_id,
        model_format=ModelFormat.SAFETENSORS,
        model=model,
        tokenizer=tokenizer,
    )
    return f"✅ **{model_id}** loaded (SafeTensors, device={model.device})."


def _load_gguf(model_id: str, gguf_filename: str) -> str:
    global _state

    if not gguf_filename:
        return "❌ Please select a GGUF filename to load the model."

    try:
        from llama_cpp import Llama
    except ImportError:
        return (
            "❌ `llama-cpp-python` package is not installed. "
            "Install: `pip install llama-cpp-python`"
        )

    model = Llama.from_pretrained(
        repo_id=model_id,
        filename=gguf_filename,
        n_ctx=2048,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0,
        verbose=False,
    )

    _state = LoadedModelState(
        model_id=model_id,
        model_format=ModelFormat.GGUF,
        model=model,
        gguf_filename=gguf_filename,
    )
    gpu_info = "GPU" if torch.cuda.is_available() else "CPU"
    return f"✅ **{model_id}** / `{gguf_filename}` loaded (GGUF, {gpu_info})."
