"""
core/memory.py - Short-term conversation context + long-term SQLite memory.
"""

import sqlite3
import json
# pyrefly: ignore [missing-import]
import chromadb
import logging
from datetime import datetime
from config import MEMORY_DB_PATH, MAX_CONTEXT_TURNS

logger = logging.getLogger("Charlie.memory")


class Memory:
    def __init__(self):
        self.db_path = MEMORY_DB_PATH
        # Persistent connection — no more open/close on every call!
        # check_same_thread=False is needed because asyncio may call from different threads.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self.chroma_client = chromadb.PersistentClient(path="data/chroma")
        
        # Create or get a "collection" (like a SQL table) named 'facts'
        self.collection = self.chroma_client.get_or_create_collection(name="facts")

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role      TEXT NOT NULL,   -- 'user' or 'assistant'
                content   TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                key       TEXT UNIQUE NOT NULL,
                value     TEXT NOT NULL,
                updated   TEXT NOT NULL
            )
        """)
        self.conn.commit()
        logger.info("Memory DB initialized at %s", self.db_path)

    def add_turn(self, user: str, assistant: str):
        """Save one conversation exchange."""
        ts = datetime.now().isoformat()
        self.conn.execute("INSERT INTO conversations (timestamp, role, content) VALUES (?, ?, ?)",
                     (ts, "user", user))
        self.conn.execute("INSERT INTO conversations (timestamp, role, content) VALUES (?, ?, ?)",
                     (ts, "assistant", assistant))
        self.conn.commit()

    def get_context(self) -> list[dict]:
        """Return last N turns as list of {role, content} dicts for LLM."""
        rows = self.conn.execute(
            "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
            (MAX_CONTEXT_TURNS * 2,)
        ).fetchall()
        # Reverse to chronological order
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def save_fact(self, key: str, value: str):
        """Store a long-term fact in SQLite and ChromaDB."""
        ts = datetime.now().isoformat()
        
        # 1. Save to SQLite (our old system, good for exact key lookups)
        self.conn.execute(
            "INSERT INTO facts (key, value, updated) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (key, value, ts)
        )
        self.conn.commit()
            
        # 2. Save to ChromaDB (for semantic/meaning search later)
        self.collection.upsert(
            documents=[value],
            metadatas=[{"key": key, "updated": ts}],
            ids=[key]
        )
        
        logger.info("Fact saved to SQLite & Chroma: %s = %s", key, value)


    def get_fact(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def get_all_facts(self) -> dict:
        rows = self.conn.execute("SELECT key, value FROM facts").fetchall()
        return {r[0]: r[1] for r in rows}

    def search_facts(self, query: str, n_results: int = 2) -> list[str]:
        """Search ChromaDB for facts semantically related to the user's query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if results and results.get("documents") and results["documents"][0]:
                return results["documents"][0]
            return []
        except Exception as e:
            logger.error("ChromaDB search failed: %s", e)
            return []

    def clear_context(self):
        """Wipe conversation history (not facts)."""
        self.conn.execute("DELETE FROM conversations")
        self.conn.commit()
        logger.info("Conversation history cleared.")

    def clear_all_facts(self):
        """Wipe all long-term facts from memory."""
        self.conn.execute("DELETE FROM facts")
        self.conn.commit()
        
        try:
            # Drop the whole collection and recreate to clear chroma
            self.chroma_client.delete_collection(name="facts")
            self.collection = self.chroma_client.create_collection(name="facts")
            logger.info("ChromaDB facts collection cleared and recreated.")
        except Exception as e:
            logger.error("Failed to clear ChromaDB: %s", e)
            
        logger.info("All long-term memory facts cleared.")

