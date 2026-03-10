"""
llm_standalone — Modular LLM Standalone Testing Package.

Modules:
  config         – App-wide constants & defaults
  model_manager  – Cache scanning, model load/unload (SafeTensors + GGUF)
  generation     – Text generation logic
  ui             – Gradio Blocks interface
"""

from .config import APP_CONFIG, ModelFormat
from .model_manager import scan_cached_models, load_model, unload_model, is_loaded
from .generation import generate_response
from .ui import build_standalone_interface

__all__ = [
    "APP_CONFIG",
    "ModelFormat",
    "scan_cached_models",
    "load_model",
    "unload_model",
    "is_loaded",
    "generate_response",
    "build_standalone_interface",
]
