"""
HuggingFace Model Downloader — download SafeTensors & GGUF models via Gradio UI.

Usage:
  python fetch_models_app.py
"""

from fetch_models import build_fetch_interface


if __name__ == "__main__":
    demo = build_fetch_interface()
    demo.launch()
