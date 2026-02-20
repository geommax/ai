"""Configuration management for LLM Server."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

from src.tuning.params import TuningParams

# ── Paths ──────────────────────────────────────────────────────────────
CONFIG_DIR = Path.home() / ".config" / "llm_server_ai"
CONFIG_FILE = CONFIG_DIR / "config.json"
DB_FILE = CONFIG_DIR / "server.db"
CACHE_DIR = Path.home() / ".cache" / "huggingface"


@dataclass
class ServerConfig:
    """Server-wide configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    active_model: str = ""
    server_was_running: bool = False
    auto_restore: bool = True
    tuning: TuningParams = field(default_factory=TuningParams)

    # ── Persistence ────────────────────────────────────────────────────
    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls) -> "ServerConfig":
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as fh:
                    data = json.load(fh)
                tuning_data = data.pop("tuning", {})
                tuning = TuningParams(**tuning_data)
                return cls(tuning=tuning, **data)
            except Exception:
                pass
        return cls()
