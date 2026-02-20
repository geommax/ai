"""HuggingFace Transformers backend — safetensors / pytorch models."""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import torch

from src.llms.backends.base import BaseBackend


class TransformersBackend(BaseBackend):
    """Run models via ``transformers.AutoModelForCausalLM``."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_id: Optional[str] = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Lifecycle ──────────────────────────────────────────────────
    def load(self, model_path: str, **kwargs: Any) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.unload()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self._device == "cuda":
            load_kwargs.update(dtype=torch.float16, device_map="auto")

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, **load_kwargs
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model_id = model_path

    def unload(self) -> None:
        if self._model is not None:
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(self._model)
            except Exception:
                pass

            try:
                self._model.tie_weights = lambda: None
            except Exception:
                pass

            try:
                for param in self._model.parameters():
                    param.data = torch.empty(0)
                for buf in self._model.buffers():
                    buf.data = torch.empty(0)
            except Exception:
                pass

            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._model_id = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Generation ─────────────────────────────────────────────────
    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded — load a model first.")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(kwargs.get("max_tokens", 512)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "top_k": int(kwargs.get("top_k", 50)),
            "repetition_penalty": float(kwargs.get("repetition_penalty", 1.1)),
            "do_sample": bool(kwargs.get("do_sample", True)),
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat_generate(self, messages: list[dict], **kwargs: Any) -> str:
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        prompt_parts.append("Assistant:")
        return self.generate("".join(prompt_parts), **kwargs)

    # ── Introspection ──────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "transformers"

    def device_info(self) -> Dict[str, str]:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "type": "CUDA",
                "name": props.name,
                "memory": f"{props.total_memory / 1024**3:.1f} GB",
            }
        return {"type": "CPU", "name": "—", "memory": "—"}
