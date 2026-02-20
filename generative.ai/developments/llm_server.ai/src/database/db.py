"""SQLite storage for API keys, usage tracking, and metadata."""

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

    # Valid key statuses
    STATUSES = ("active", "revoked", "deleted", "expired")

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS key_usage (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id            INTEGER NOT NULL,
                    endpoint          TEXT    NOT NULL,
                    prompt_tokens     INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens      INTEGER DEFAULT 0,
                    created_at        TEXT    NOT NULL,
                    FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
                )
                """
            )
            # Index for fast per-key lookups
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_key_usage_key_id
                ON key_usage(key_id)
                """
            )

            # ── Migrations ─────────────────────────────────────────
            cols = [
                r[1]
                for r in conn.execute(
                    "PRAGMA table_info(api_keys)"
                ).fetchall()
            ]
            # Add deleted_at if missing
            if "deleted_at" not in cols:
                conn.execute(
                    "ALTER TABLE api_keys "
                    "ADD COLUMN deleted_at TEXT DEFAULT NULL"
                )
            # Migrate is_active (int) → status (text)
            if "status" not in cols:
                conn.execute(
                    "ALTER TABLE api_keys "
                    "ADD COLUMN status TEXT DEFAULT 'active'"
                )
                # Back-fill from existing is_active values
                conn.execute(
                    "UPDATE api_keys SET status = CASE "
                    "WHEN is_active = 1 THEN 'active' "
                    "WHEN is_active = -1 THEN 'deleted' "
                    "ELSE 'revoked' END"
                )
            conn.commit()
            # Unique constraint on key name
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_name "
                "ON api_keys(name)"
            )
            conn.commit()

    # ── CRUD ───────────────────────────────────────────────────────────
    def add_key(self, name: str, key: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            dup = conn.execute(
                "SELECT 1 FROM api_keys WHERE name = ?", (name,)
            ).fetchone()
            if dup:
                raise ValueError(f"Key name '{name}' already exists")
            conn.execute(
                "INSERT INTO api_keys (name, key, created_at, status) "
                "VALUES (?, ?, ?, 'active')",
                (name, key, datetime.now().isoformat()),
            )
            conn.commit()

    def get_keys(self, status_filter: str = "all") -> List[Dict]:
        """Return API keys filtered by status.

        *status_filter*: ``"all"`` (non-deleted), ``"active"``,
        ``"revoked"``, ``"expired"``, ``"deleted"``, or ``"every"``.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status_filter in self.STATUSES:
                rows = conn.execute(
                    "SELECT * FROM api_keys WHERE status = ? "
                    "ORDER BY created_at DESC",
                    (status_filter,),
                ).fetchall()
            elif status_filter == "every":
                rows = conn.execute(
                    "SELECT * FROM api_keys ORDER BY created_at DESC"
                ).fetchall()
            else:  # "all" — everything except deleted
                rows = conn.execute(
                    "SELECT * FROM api_keys WHERE status != 'deleted' "
                    "ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def set_key_status(self, key_id: int, status: str) -> None:
        """Change a key's status to *status* (active/revoked/deleted/expired)."""
        if status not in self.STATUSES:
            raise ValueError(f"Invalid status '{status}', must be one of {self.STATUSES}")
        with sqlite3.connect(self.db_path) as conn:
            extra = ""
            params: list = [status]
            if status == "deleted":
                extra = ", deleted_at = ?"
                params.append(datetime.now().isoformat())
            else:
                extra = ", deleted_at = NULL"
            params.append(key_id)
            conn.execute(
                f"UPDATE api_keys SET status = ?{extra} WHERE id = ?",
                params,
            )
            conn.commit()

    # ── Legacy convenience wrappers (thin) ─────────────────────────
    def revoke_key(self, key_id: int) -> None:
        self.set_key_status(key_id, "revoked")

    def activate_key(self, key_id: int) -> None:
        self.set_key_status(key_id, "active")

    def delete_key(self, key_id: int) -> None:
        self.set_key_status(key_id, "deleted")

    def validate_key(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM api_keys WHERE key = ? AND status = 'active'",
                (key,),
            ).fetchone()
            return row is not None

    def validate_key_get_id(self, key: str) -> Optional[int]:
        """Validate an API key; return its id if active, else None."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM api_keys WHERE key = ? AND status = 'active'",
                (key,),
            ).fetchone()
            return row[0] if row else None

    def key_count(self, active_only: bool = True) -> int:
        with sqlite3.connect(self.db_path) as conn:
            if active_only:
                q = "SELECT COUNT(*) FROM api_keys WHERE status = 'active'"
            else:
                q = "SELECT COUNT(*) FROM api_keys WHERE status != 'deleted'"
            return conn.execute(q).fetchone()[0]

    # ── Usage tracking ─────────────────────────────────────────────────
    def record_usage(
        self,
        key_id: int,
        endpoint: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Record a single API call for usage metering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO key_usage
                   (key_id, endpoint, prompt_tokens, completion_tokens,
                    total_tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    key_id,
                    endpoint,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def get_key_usage(self, key_id: Optional[int] = None) -> List[Dict]:
        """Return per-key aggregated usage stats.

        If *key_id* is given, return stats for that key only; otherwise
        return stats for every key.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if key_id is not None:
                rows = conn.execute(
                    """SELECT k.id, k.name,
                              COUNT(u.id) AS total_calls,
                              COALESCE(SUM(u.prompt_tokens), 0)     AS prompt_tokens,
                              COALESCE(SUM(u.completion_tokens), 0) AS completion_tokens,
                              COALESCE(SUM(u.total_tokens), 0)      AS total_tokens,
                              MAX(u.created_at)                     AS last_used
                       FROM api_keys k
                       LEFT JOIN key_usage u ON k.id = u.key_id
                       WHERE k.id = ?
                       GROUP BY k.id""",
                    (key_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT k.id, k.name,
                              COUNT(u.id) AS total_calls,
                              COALESCE(SUM(u.prompt_tokens), 0)     AS prompt_tokens,
                              COALESCE(SUM(u.completion_tokens), 0) AS completion_tokens,
                              COALESCE(SUM(u.total_tokens), 0)      AS total_tokens,
                              MAX(u.created_at)                     AS last_used
                       FROM api_keys k
                       LEFT JOIN key_usage u ON k.id = u.key_id
                       GROUP BY k.id
                       ORDER BY total_calls DESC"""
                ).fetchall()
            return [dict(r) for r in rows]

    def get_key_usage_history(self, key_id: int) -> List[Dict]:
        """Return individual API-call records for *key_id* (newest first)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT endpoint, prompt_tokens, completion_tokens,
                          total_tokens, created_at
                   FROM key_usage
                   WHERE key_id = ?
                   ORDER BY created_at DESC
                   LIMIT 100""",
                (key_id,),
            ).fetchall()
            return [dict(r) for r in rows]
