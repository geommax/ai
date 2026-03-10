"""
LLM Standalone (Without RAG Pipeline)
A standalone script to test the LLM's raw knowledge before integrating a RAG pipeline.

Purpose:
  - Test the LLM independently to check whether it has domain knowledge
  - Evaluate whether RAG augmentation is needed
  - Tune generation parameters (temperature, top_p, top_k, etc.)
  - Support both SafeTensors (Transformers) and GGUF (llama.cpp) formats
  - Lazy loading: model is loaded only after UI is up via the Load Model button

Usage:
  python main.py
"""

from llm_standalone import build_standalone_interface


if __name__ == "__main__":
    demo = build_standalone_interface()
    demo.launch()
