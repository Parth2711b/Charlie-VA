"""
core/memory.py — Short-term conversation context + long-term SQLite memory.
"""

import sqlite3
import json
import logging
from datetime import datetime
from config import MEMORY_DB_PATH, MAX_CONTEXT_TURNS

logger = logging.getLogger("Charlie.memory")


class Memory:
    def __init__(self):
        self.db_path = MEMORY_DB_PATH
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    role      TEXT NOT NULL,   -- 'user' or 'assistant'
                    content   TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    key       TEXT UNIQUE NOT NULL,
                    value     TEXT NOT NULL,
                    updated   TEXT NOT NULL
                )
            """)
            conn.commit()
        logger.info("Memory DB initialized at %s", self.db_path)

    def add_turn(self, user: str, assistant: str):
        """Save one conversation exchange."""
        ts = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO conversations (timestamp, role, content) VALUES (?, ?, ?)",
                         (ts, "user", user))
            conn.execute("INSERT INTO conversations (timestamp, role, content) VALUES (?, ?, ?)",
                         (ts, "assistant", assistant))
            conn.commit()

    def get_context(self) -> list[dict]:
        """Return last N turns as list of {role, content} dicts for LLM."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
                (MAX_CONTEXT_TURNS * 2,)
            ).fetchall()
        # Reverse to chronological order
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def save_fact(self, key: str, value: str):
        """Store a long-term fact (e.g. user preferences, names)."""
        ts = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO facts (key, value, updated) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
                (key, value, ts)
            )
            conn.commit()
        logger.info("Fact saved: %s = %s", key, value)

    def get_fact(self, key: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def get_all_facts(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT key, value FROM facts").fetchall()
        return {r[0]: r[1] for r in rows}

    def clear_context(self):
        """Wipe conversation history (not facts)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversations")
            conn.commit()
        logger.info("Conversation history cleared.")
