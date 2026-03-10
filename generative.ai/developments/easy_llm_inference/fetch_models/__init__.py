"""
fetch_models — HuggingFace Model Downloader Package.

Modules:
  downloader  – Download logic for SafeTensors & GGUF
  ui          – Gradio Blocks interface
"""

from .downloader import download_safetensors, download_gguf, list_gguf_files
from .ui import build_fetch_interface

__all__ = [
    "download_safetensors",
    "download_gguf",
    "list_gguf_files",
    "build_fetch_interface",
]
