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
