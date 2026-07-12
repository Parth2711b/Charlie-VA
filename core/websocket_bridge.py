"""
core/websocket_bridge.py - WebSocket server bridging Charlie (Python) and Dashboard (Browser).
Runs as a background task alongside the main assistant loop.

Dashboard connects to ws://localhost:8765
Messages from Charlie → Dashboard: JSON objects
Messages from Dashboard → Charlie: JSON objects
"""

import asyncio
import json
import logging
import websockets
from websockets.server import ServerConnection

logger = logging.getLogger("charlie.ws_bridge")

# ── Connected clients ──────────────────────────────────────────────────────────
_clients: set[ServerConnection] = set()


def has_clients() -> bool:
    """Check if any dashboard clients are connected. Use this instead of accessing _clients directly."""
    return len(_clients) > 0


# ── Broadcast to all dashboard clients ────────────────────────────────────────

async def broadcast(message: dict):
    """Send a message to all connected dashboard clients."""
    if not _clients:
        return
    data = json.dumps(message)
    disconnected = set()
    for client in _clients:
        try:
            await client.send(data)
        except Exception:
            disconnected.add(client)
    _clients.difference_update(disconnected)


# ── Convenience senders ───────────────────────────────────────────────────────

async def send_heard(text: str):
    """Tell dashboard what the user said."""
    await broadcast({"type": "heard", "text": text})


async def send_response(text: str):
    """Tell dashboard what Charlie responded."""
    await broadcast({"type": "response", "text": text})


async def send_audio(base64_data: str):
    """Send base64 encoded audio to the dashboard for playback."""
    await broadcast({"type": "audio", "data": base64_data})


async def send_stop_audio():
    """Tell dashboard to instantly stop audio playback."""
    await broadcast({"type": "stop_audio"})


async def send_status(stt: str = "-", llm: str = "-", mem: str = "-"):
    """Update system status panel on dashboard."""
    await broadcast({"type": "status", "stt": stt, "llm": llm, "mem": mem})


async def send_load_url(url: str, mode: str = "url"):
    """Tell dashboard to load a URL in the content frame."""
    await broadcast({"type": "load_url", "url": url, "mode": mode})


async def send_focus_panel(panel: str):
    """Tell dashboard to focus a panel: news/content/seismic/charlie."""
    await broadcast({"type": "focus_panel", "panel": panel})


async def send_memory(facts: list[str]):
    """Send memory facts to dashboard."""
    await broadcast({"type": "memory", "facts": facts})


async def send_interview_mode(active: bool):
    """Tell dashboard to show/hide the interview mode indicator."""
    await broadcast({"type": "interview_mode", "active": active})


# ── Incoming message handler ──────────────────────────────────────────────────

# Callback - set by assistant.py to handle text input from dashboard
_on_text_input = None


def set_text_input_handler(handler):
    """Register callback for text messages from dashboard."""
    global _on_text_input
    _on_text_input = handler


async def _handle_client(websocket: ServerConnection):
    """Handle a single dashboard client connection."""
    _clients.add(websocket)
    client_addr = websocket.remote_address
    logger.info("Dashboard connected: %s", client_addr)

    # Send initial status
    await send_status(stt="READY", llm="READY", mem="READY")

    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
                msg_type = data.get("type")

                if msg_type == "text_input" and _on_text_input:
                    text = data.get("text", "").strip()
                    if text:
                        logger.info("Dashboard text input: %s", text)
                        asyncio.create_task(_on_text_input(text))

                elif msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

            except json.JSONDecodeError:
                logger.warning("Invalid JSON from dashboard: %s", raw)

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning("Dashboard disconnected unexpectedly: %s", e)
    finally:
        _clients.discard(websocket)
        logger.info("Dashboard disconnected: %s", client_addr)


# ── Server startup ────────────────────────────────────────────────────────────

async def start_server(host: str = "localhost", port: int = 8765):
    """Start the WebSocket server. Call this as an asyncio task."""
    logger.info("WebSocket bridge starting on ws://%s:%d", host, port)
    async with websockets.serve(_handle_client, host, port):
        logger.info("WebSocket bridge ready - dashboard can connect.")
        await asyncio.Future()  # run forever