"""
core/websocket_bridge.py - WebSocket server bridging Charlie (Python) and Dashboard (Browser).
Multi-Tenant Architecture using contextvars.
"""

import asyncio
import json
import logging
import websockets
import contextvars
from typing import Dict, Set

logger = logging.getLogger("charlie.ws_bridge")

# ── Multi-Tenant State ────────────────────────────────────────────────────────
# Mapping of user_id to a set of active websocket connections
_clients: Dict[int, Set] = {}

# Context variable to track the current user for the active asyncio task
current_user_id: contextvars.ContextVar[int | None] = contextvars.ContextVar("current_user_id", default=None)

def has_clients() -> bool:
    """Check if the CURRENT user has any active dashboard clients connected (defaults to user 1 for local commands)."""
    uid = current_user_id.get() or 1
    return bool(uid and _clients.get(uid))

def has_any_clients() -> bool:
    """Check if ANY dashboard clients are connected at all."""
    return any(_clients.values())


# ── Private Routing (Emit to specific user) ───────────────────────────────────

async def emit(message: dict, target_user_id: int | None = None):
    """Send a message to a specific user (defaults to current context user, or user 1 for local commands)."""
    uid = target_user_id or current_user_id.get() or 1
    if not uid or uid not in _clients or not _clients[uid]:
        return
        
    data = json.dumps(message)
    disconnected = set()
    for client in _clients[uid]:
        try:
            await client.send(data)
        except Exception:
            disconnected.add(client)
    _clients[uid].difference_update(disconnected)


# ── Convenience senders (Automatically route to current user context!) ────────

async def send_heard(text: str, target_user_id: int | None = None):
    await emit({"type": "heard", "text": text}, target_user_id=target_user_id)

async def send_response(text: str, target_user_id: int | None = None):
    await emit({"type": "response", "text": text}, target_user_id=target_user_id)

async def send_audio(base64_data: str, target_user_id: int | None = None):
    await emit({"type": "audio", "data": base64_data}, target_user_id=target_user_id)

async def send_stop_audio(target_user_id: int | None = None):
    await emit({"type": "stop_audio"}, target_user_id=target_user_id)

async def send_listening(state: bool, target_user_id: int | None = None):
    await emit({"type": "listening", "state": state}, target_user_id=target_user_id)

async def send_status(stt: str = "-", llm: str = "-", mem: str = "-", target_user_id: int | None = None):
    await emit({"type": "status", "stt": stt, "llm": llm, "mem": mem}, target_user_id=target_user_id)

async def send_load_url(url: str, mode: str = "url", target_user_id: int | None = None):
    await emit({"type": "load_url", "url": url, "mode": mode}, target_user_id=target_user_id)

async def send_focus_panel(panel: str, target_user_id: int | None = None):
    await emit({"type": "focus_panel", "panel": panel}, target_user_id=target_user_id)

async def send_memory(facts: list[str], target_user_id: int | None = None):
    await emit({"type": "memory", "facts": facts}, target_user_id=target_user_id)

async def send_interview_mode(active: bool, target_user_id: int | None = None):
    await emit({"type": "interview_mode", "active": active}, target_user_id=target_user_id)


# ── Callbacks ─────────────────────────────────────────────────────────────────
_on_text_input = None
_on_audio_input = None
_on_barge_in = None

def set_text_input_handler(handler):
    global _on_text_input
    _on_text_input = handler

def set_audio_input_handler(handler):
    global _on_audio_input
    _on_audio_input = handler

def set_barge_in_handler(handler):
    global _on_barge_in
    _on_barge_in = handler


# ── Connection Handler ────────────────────────────────────────────────────────

async def _handle_client(websocket):
    """Handle a single dashboard client connection."""
    client_addr = websocket.remote_address
    logger.info("Dashboard connected: %s", client_addr)
    
    # Each websocket starts unauthenticated
    auth_user_id = None 

    try:
        async for raw in websocket:
            # We must set the context variable on EVERY message so downstream tasks know the user!
            if auth_user_id:
                current_user_id.set(auth_user_id)
                
            if isinstance(raw, bytes):
                if auth_user_id and _on_audio_input:
                    # Context is automatically copied to the new task
                    asyncio.create_task(_on_audio_input(raw))
                continue

            try:
                data = json.loads(raw)
                msg_type = data.get("type")

                # Phase 3 Authentication / Login
                if msg_type == "authenticate":
                    # For now, just trust the ID passed from the dashboard
                    auth_user_id = data.get("user_id")
                    if auth_user_id:
                        if auth_user_id not in _clients:
                            _clients[auth_user_id] = set()
                        _clients[auth_user_id].add(websocket)
                        current_user_id.set(auth_user_id)
                        await send_status(stt="READY", llm="READY", mem="READY")
                        logger.info("Client %s authenticated as User %s", client_addr, auth_user_id)
                        
                        from core.memory import Memory
                        try:
                            # Use Memory DB instead of Schedule DB!
                            history = Memory().get_context(auth_user_id)
                            await websocket.send(json.dumps({"type": "chat_history", "history": history}))
                        except Exception as e:
                            logger.error("Failed to load chat history: %s", e)
                            
                    continue

                if msg_type == "login":
                    username = data.get("username")
                    password = data.get("password")
                    from core.schedule_db import schedule_db
                    from core.models import User
                    with schedule_db.SessionLocal() as db:
                        user = db.query(User).filter(User.username == username).first()
                        if user and user.hashed_password == password:
                            await websocket.send(json.dumps({"type": "login_result", "success": True, "user_id": user.id}))
                        else:
                            await websocket.send(json.dumps({"type": "login_result", "success": False, "error": "Invalid credentials"}))
                    continue
                    
                if msg_type == "register":
                    username = data.get("username")
                    password = data.get("password")
                    from core.schedule_db import schedule_db
                    from core.models import User
                    with schedule_db.SessionLocal() as db:
                        if db.query(User).filter(User.username == username).first():
                            await websocket.send(json.dumps({"type": "register_result", "success": False, "error": "Username taken"}))
                        else:
                            new_user = User(username=username, hashed_password=password)
                            db.add(new_user)
                            db.commit()
                            db.refresh(new_user)
                            await websocket.send(json.dumps({"type": "register_result", "success": True, "user_id": new_user.id}))
                    continue

                if not auth_user_id:
                    logger.warning("Unauthenticated message received: %s from %s", msg_type, client_addr)
                    continue
                    
                # All commands below are executed within the authenticated context
                if msg_type == "text_input" and _on_text_input:
                    text = data.get("text", "").strip()
                    if text:
                        logger.info("[User %s] Dashboard text: %s", auth_user_id, text)
                        asyncio.create_task(_on_text_input(text))

                elif msg_type == "barge_in":
                    if _on_barge_in:
                        _on_barge_in()

                elif msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

                elif msg_type == "get_schedule":
                    from core.schedule_db import schedule_db
                    events = schedule_db.get_upcoming_events(auth_user_id)
                    await websocket.send(json.dumps({"type": "schedule_data", "events": events}))
                    
                elif msg_type == "get_flights":
                    lat, lon = data.get("lat"), data.get("lon")
                    import httpx
                    url = f"https://opensky-network.org/api/states/all?lamin={lat-5}&lomin={lon-5}&lamax={lat+5}&lomax={lon+5}"
                    try:
                        async with httpx.AsyncClient() as client:
                            resp = await client.get(url, timeout=10.0)
                            if resp.status_code == 200:
                                await websocket.send(json.dumps({"type": "flight_data", "data": resp.json()}))
                            else:
                                await websocket.send(json.dumps({"type": "flight_data", "data": None}))
                    except Exception as e:
                        logger.error("OpenSky API error: %s", e)
                        await websocket.send(json.dumps({"type": "flight_data", "data": None}))
                    
                elif msg_type == "mark_schedule_done":
                    from core.schedule_db import schedule_db
                    ev_id = data.get("id")
                    if ev_id is not None:
                        schedule_db.mark_done(ev_id)
                        events = schedule_db.get_upcoming_events(auth_user_id)
                        await websocket.send(json.dumps({"type": "schedule_data", "events": events}))

            except json.JSONDecodeError:
                logger.warning("Invalid JSON from dashboard: %s", raw)

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning("Dashboard disconnected unexpectedly: %s", e)
    finally:
        if auth_user_id and auth_user_id in _clients:
            _clients[auth_user_id].discard(websocket)
        logger.info("Dashboard disconnected: %s", client_addr)


# ── Server startup ────────────────────────────────────────────────────────────

async def start_server(host: str = "localhost", port: int = 8765):
    """Start the WebSocket server. Call this as an asyncio task."""
    logger.info("WebSocket bridge starting on ws://%s:%d", host, port)
    async with websockets.serve(_handle_client, host, port):
        logger.info("WebSocket bridge ready - dashboard can connect.")
        await asyncio.Future()  # run forever