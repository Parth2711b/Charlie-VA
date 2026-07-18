"""
core/schedule_db.py
Database layer for scheduled tasks and reminders via SQLAlchemy.
"""

import logging
from datetime import datetime
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.models import Base, ScheduleEvent

logger = logging.getLogger("Charlie.schedule_db")

class ScheduleDB:
    def __init__(self, db_path: str = "./data/schedule.sqlite3"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def add_event(self, user_id: int, event_text: str, scheduled_time: datetime) -> int | None:
        ts = datetime.now().isoformat()
        with self.SessionLocal() as db:
            event = ScheduleEvent(
                user_id=user_id,
                event_text=event_text,
                scheduled_time=scheduled_time.isoformat(),
                status='pending',
                created_at=ts
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            return event.id

    def get_upcoming_events(self, user_id: int) -> list[dict]:
        with self.SessionLocal() as db:
            events = db.query(ScheduleEvent).filter(
                ScheduleEvent.user_id == user_id, 
                ScheduleEvent.status == 'pending'
            ).order_by(ScheduleEvent.scheduled_time.asc()).all()
            return [{"id": e.id, "event_text": e.event_text, "scheduled_time": e.scheduled_time, "status": e.status} for e in events]
        
    def get_todays_schedule(self, user_id: int) -> list[dict]:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
        with self.SessionLocal() as db:
            events = db.query(ScheduleEvent).filter(
                ScheduleEvent.user_id == user_id,
                ScheduleEvent.scheduled_time >= today_start,
                ScheduleEvent.scheduled_time <= today_end
            ).order_by(ScheduleEvent.scheduled_time.asc()).all()
            return [{"id": e.id, "event_text": e.event_text, "scheduled_time": e.scheduled_time, "status": e.status} for e in events]

    def get_due_events(self) -> list[dict]:
        """Note: This fetches ALL due events across ALL users (useful for background polling loop)."""
        now = datetime.now().isoformat()
        with self.SessionLocal() as db:
            events = db.query(ScheduleEvent).filter(
                ScheduleEvent.status == 'pending',
                ScheduleEvent.scheduled_time <= now
            ).all()
            return [{"id": e.id, "user_id": e.user_id, "event_text": e.event_text, "scheduled_time": e.scheduled_time} for e in events]

    def mark_done(self, event_id: int):
        with self.SessionLocal() as db:
            event = db.query(ScheduleEvent).filter(ScheduleEvent.id == event_id).first()
            if event:
                event.status = 'completed'
                db.commit()
        
    def delete_event(self, event_id: int):
        with self.SessionLocal() as db:
            db.query(ScheduleEvent).filter(ScheduleEvent.id == event_id).delete()
            db.commit()

schedule_db = ScheduleDB()
