"""Inference engine — load models and generate text."""

from __future__ import annotations

import gc

import torch
from typing import Any, Dict, Optional


class InferenceEngine:
    """Wraps a single ``transformers`` causal-LM for text generation."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_id: Optional[str] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load / Unload ──────────────────────────────────────────────────
    def load_model(self, model_id: str) -> None:
        """Load *model_id* (from local cache or hub) into memory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.unload_model()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self.device == "cuda":
            load_kwargs.update(dtype=torch.float16, device_map="auto")

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_id = model_id

    def unload_model(self) -> None:
        """Release model, tokenizer and reclaim GPU memory."""
        if self.model is not None:
            # If accelerate device_map was used, detach dispatch hooks first
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(self.model)
            except Exception:
                pass

            # Move model to CPU before deleting (releases CUDA tensors)
            try:
                self.model.to("cpu")
            except Exception:
                pass

            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.model_id = None

        # Force Python GC to break reference cycles, then free CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ── Generate ───────────────────────────────────────────────────────
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text continuation for *prompt*."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded — load a model first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(kwargs.get("max_tokens", 512)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "top_k": int(kwargs.get("top_k", 50)),
            "repetition_penalty": float(kwargs.get("repetition_penalty", 1.1)),
            "do_sample": bool(kwargs.get("do_sample", True)),
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Chat-style helper ──────────────────────────────────────────────
    def chat_generate(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> str:
        """Build a prompt from chat *messages* and generate a response."""
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

    # ── Introspection ──────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def device_info(self) -> Dict[str, str]:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "type": "CUDA",
                "name": props.name,
                "memory": f"{props.total_memory / 1024**3:.1f} GB",
            }
        return {"type": "CPU", "name": "—", "memory": "—"}
