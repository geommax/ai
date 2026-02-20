"""LLM Server.AI — Local LLM Server with TUI

Modular structure:
  src.ui        — Textual TUI application and screens
  src.llms      — Model management and inference engine
  src.apis      — API key generation and FastAPI server
  src.database  — SQLite storage for API keys
  src.tuning    — Generation hyper-parameter management
  src.config    — Shared configuration (ServerConfig, paths)
"""
