"""
core/memory.py - Short-term conversation context + long-term SQLite memory via SQLAlchemy.
"""

import json
# pyrefly: ignore [missing-import]
import chromadb
import logging
from datetime import datetime
from config import MEMORY_DB_PATH, MAX_CONTEXT_TURNS

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.models import Base, User, ConversationTurn, Fact

logger = logging.getLogger("Charlie.memory")


class Memory:
    def __init__(self):
        self.db_path = MEMORY_DB_PATH
        # Create SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self.chroma_client = chromadb.PersistentClient(path="data/chroma")
        
        # Create or get a "collection" (like a SQL table) named 'facts'
        self.collection = self.chroma_client.get_or_create_collection(name="facts")
        logger.info("Memory DB initialized (SQLAlchemy) at %s", self.db_path)

    def add_turn(self, user_id: int, user: str, assistant: str):
        """Save one conversation exchange."""
        ts = datetime.now().isoformat()
        with self.SessionLocal() as db:
            turn_user = ConversationTurn(user_id=user_id, timestamp=ts, role="user", content=user)
            turn_assistant = ConversationTurn(user_id=user_id, timestamp=ts, role="assistant", content=assistant)
            db.add(turn_user)
            db.add(turn_assistant)
            db.commit()

    def get_context(self, user_id: int) -> list[dict]:
        """Return last N turns as list of {role, content} dicts for LLM."""
        with self.SessionLocal() as db:
            rows = db.query(ConversationTurn).filter(ConversationTurn.user_id == user_id).order_by(ConversationTurn.id.desc()).limit(MAX_CONTEXT_TURNS * 2).all()
        # Reverse to chronological order
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    def save_fact(self, user_id: int, key: str, value: str):
        """Store a long-term fact in SQLite and ChromaDB."""
        ts = datetime.now().isoformat()
        
        # 1. Save to SQLAlchemy
        with self.SessionLocal() as db:
            fact = db.query(Fact).filter(Fact.user_id == user_id, Fact.key == key).first()
            if fact:
                fact.value = value
                fact.updated = ts
            else:
                fact = Fact(user_id=user_id, key=key, value=value, updated=ts)
                db.add(fact)
            db.commit()
            
        # 2. Save to ChromaDB (composite ID so users don't overwrite each other)
        composite_id = f"{user_id}_{key}"
        self.collection.upsert(
            documents=[value],
            metadatas=[{"key": key, "updated": ts, "user_id": user_id}],
            ids=[composite_id]
        )
        
        logger.info("Fact saved for user %s: %s = %s", user_id, key, value)


    def get_fact(self, user_id: int, key: str) -> str | None:
        with self.SessionLocal() as db:
            fact = db.query(Fact).filter(Fact.user_id == user_id, Fact.key == key).first()
            return fact.value if fact else None

    def get_all_facts(self, user_id: int) -> dict:
        with self.SessionLocal() as db:
            facts = db.query(Fact).filter(Fact.user_id == user_id).all()
            return {f.key: f.value for f in facts}

    def search_facts(self, user_id: int, query: str, n_results: int = 2) -> list[str]:
        """Search ChromaDB for facts semantically related to the user's query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"user_id": user_id}
            )
            if results and results.get("documents") and results["documents"][0]:
                return results["documents"][0]
            return []
        except Exception as e:
            logger.error("ChromaDB search failed: %s", e)
            return []

    def clear_context(self, user_id: int):
        """Wipe conversation history (not facts)."""
        with self.SessionLocal() as db:
            db.query(ConversationTurn).filter(ConversationTurn.user_id == user_id).delete()
            db.commit()
        logger.info("Conversation history cleared for user %s.", user_id)

    def clear_all_facts(self, user_id: int):
        """Wipe all long-term facts from memory."""
        with self.SessionLocal() as db:
            db.query(Fact).filter(Fact.user_id == user_id).delete()
            db.commit()
        
        try:
            self.collection.delete(where={"user_id": user_id})
            logger.info("ChromaDB facts deleted for user %s.", user_id)
        except Exception as e:
            logger.error("Failed to clear ChromaDB for user %s: %s", user_id, e)
            
        logger.info("All long-term memory facts cleared for user %s.", user_id)

