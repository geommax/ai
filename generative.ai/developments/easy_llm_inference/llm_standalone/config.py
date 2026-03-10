"""
Configuration — default values & constants for LLM Standalone.
"""

from dataclasses import dataclass, field
from enum import Enum


class ModelFormat(str, Enum):
    """Supported model weight formats."""
    SAFETENSORS = "SafeTensors (Transformers)"
    GGUF = "GGUF (llama.cpp)"


@dataclass
class GenerationDefaults:
    """Default generation hyper-parameters."""
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512


# ── Prompt template presets ──────────────────────────────────────────────

PROMPT_TEMPLATES: dict[str, str] = {
    "Auto (Chat Template)": "",                             # use tokenizer.apply_chat_template / chat API
    "ChatML": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "Llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "Alpaca": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{system}\n\n"
        "### Input:\n{prompt}\n\n"
        "### Response:\n"
    ),
    "Vicuna": (
        "{system}\n\n"
        "USER: {prompt}\n"
        "ASSISTANT: "
    ),
    "Phi": (
        "<|system|>\n{system}<|end|>\n"
        "<|user|>\n{prompt}<|end|>\n"
        "<|assistant|>\n"
    ),
    "Custom": "{system}\n\n{prompt}",                       # user-editable
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class AppConfig:
    """Top-level application settings."""
    default_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    default_gguf_filename: str = ""
    default_format: ModelFormat = ModelFormat.SAFETENSORS
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)
    server_port: int = 7860


# Singleton config instance
APP_CONFIG = AppConfig()
