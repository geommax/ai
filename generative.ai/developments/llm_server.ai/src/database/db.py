"""SQLite storage for API keys and metadata."""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class Database:
    """Thin wrapper around a SQLite database for API-key management."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Schema ─────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT    NOT NULL,
                    key        TEXT    UNIQUE NOT NULL,
                    created_at TEXT    NOT NULL,
                    is_active  INTEGER DEFAULT 1
                )
                """
            )
            conn.commit()

    # ── CRUD ───────────────────────────────────────────────────────────
    def add_key(self, name: str, key: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO api_keys (name, key, created_at) VALUES (?, ?, ?)",
                (name, key, datetime.now().isoformat()),
            )
            conn.commit()

    def get_keys(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_active_keys(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE is_active = 1 ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def revoke_key(self, key_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,)
            )
            conn.commit()

    def activate_key(self, key_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE api_keys SET is_active = 1 WHERE id = ?", (key_id,)
            )
            conn.commit()

    def delete_key(self, key_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            conn.commit()

    def validate_key(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM api_keys WHERE key = ? AND is_active = 1",
                (key,),
            ).fetchone()
            return row is not None

    def key_count(self, active_only: bool = True) -> int:
        with sqlite3.connect(self.db_path) as conn:
            q = "SELECT COUNT(*) FROM api_keys"
            if active_only:
                q += " WHERE is_active = 1"
            return conn.execute(q).fetchone()[0]
