"""
handlers/notes.py - Save and retrieve notes. Stored in data/notes.json.
Offline, instant. Shows notes in the Charlie panel on the dashboard.
"""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger("Charlie.handler.notes")

NOTES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "notes.json")


def _load_notes() -> list[dict]:
    if not os.path.exists(NOTES_PATH):
        return []
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_notes(notes: list[dict]):
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


async def handle(text: str) -> str:
    import re
    text_lower = text.lower()

    # ── Read notes ──
    if any(w in text_lower for w in ["show my notes", "read notes", "what are my notes", "list notes"]):
        notes = _load_notes()
        if not notes:
            return "You have no saved notes."
        from core import websocket_bridge as ws
        await ws.broadcast({"type": "show_notes", "notes": [n["text"] for n in notes[-10:]]})
        latest = notes[-3:]
        summary = ". ".join(n["text"] for n in latest)
        return f"You have {len(notes)} notes. Last ones: {summary}"

    # ── Delete all notes ──
    if any(w in text_lower for w in ["clear notes", "delete all notes", "wipe notes"]):
        _save_notes([])
        from core import websocket_bridge as ws
        await ws.broadcast({"type": "show_notes", "notes": []})
        return "All notes deleted."

    # ── Save note ──
    patterns = [
        r"(?:save a note|take a note|note|remember|write down)[:\s]+(.+)",
        r"(?:make a note|add a note)[:\s]+(.+)",
        r"note that (.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            note_text = m.group(1).strip()
            notes = _load_notes()
            notes.append({
                "text": note_text,
                "timestamp": datetime.now().isoformat()
            })
            _save_notes(notes)
            logger.info("Note saved: %s", note_text)
            from core import websocket_bridge as ws
            await ws.broadcast({"type": "show_notes", "notes": [n["text"] for n in notes[-10:]]})
            return f"Got it, I've saved your note: '{note_text}'"

    return "What should I note down? Say something like 'save a note: buy milk'."
