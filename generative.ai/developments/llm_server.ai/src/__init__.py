"""LLM Server.AI — Local LLM Server with TUI

Architecture:  daemon (backend) + TUI (frontend client)

Modular structure:
  src.daemon    — Background daemon process + Unix-socket client
  src.ui        — Textual TUI application and screens (frontend only)
  src.llms      — Model management and inference engine
  src.apis      — API key generation and FastAPI server
  src.database  — SQLite storage for API keys
  src.tuning    — Generation hyper-parameter management
  src.config    — Shared configuration (ServerConfig, paths)
"""
