"""
core/models.py - SQLAlchemy ORM Definitions for Multi-Tenant Database
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships (cascade delete so if a user is deleted, their data is wiped)
    conversations = relationship("ConversationTurn", back_populates="user", cascade="all, delete-orphan")
    facts = relationship("Fact", back_populates="user", cascade="all, delete-orphan")
    events = relationship("ScheduleEvent", back_populates="user", cascade="all, delete-orphan")


class ConversationTurn(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(String, nullable=False)
    role = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    
    user = relationship("User", back_populates="conversations")


class Fact(Base):
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key = Column(String, nullable=False)  # Note: No longer UNIQUE globally, only unique per user (handled in app logic)
    value = Column(Text, nullable=False)
    updated = Column(String, nullable=False)
    
    user = relationship("User", back_populates="facts")


class ScheduleEvent(Base):
    __tablename__ = "schedule"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    event_text = Column(String, nullable=False)
    scheduled_time = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(String, nullable=False)
    
    user = relationship("User", back_populates="events")
