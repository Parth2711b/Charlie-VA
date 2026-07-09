"""
handlers/timer.py — Set countdown timers displayed on the dashboard.
Sends timer_start to dashboard, which handles the visual countdown.
"""

import re
import logging
import asyncio

logger = logging.getLogger("Charlie.handler.timer")


def _parse_duration(text: str) -> int | None:
    """Parse duration in seconds from spoken command. Returns None if not found."""
    text_lower = text.lower()
    total = 0

    patterns = [
        (r"(\d+)\s*hour", 3600),
        (r"(\d+)\s*minute", 60),
        (r"(\d+)\s*second", 1),
        (r"(\d+)\s*min",    60),
        (r"(\d+)\s*sec",    1),
        (r"(\d+)\s*hr",     3600),
    ]
    found = False
    for pat, mult in patterns:
        m = re.search(pat, text_lower)
        if m:
            total += int(m.group(1)) * mult
            found = True

    return total if found else None


async def handle(text: str) -> str:
    text_lower = text.lower()

    # Cancel timer
    if "cancel" in text_lower or "stop timer" in text_lower:
        from core import websocket_bridge as ws
        await ws.broadcast({"type": "timer_cancel"})
        return "Timer cancelled."

    duration = _parse_duration(text)
    if not duration:
        return "How long should the timer be? Say something like 'set a 5 minute timer'."

    if duration > 86400:
        return "That's a very long timer. Maximum is 24 hours."

    # Human-readable label
    h, rem = divmod(duration, 3600)
    m, s   = divmod(rem, 60)
    parts  = []
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s: parts.append(f"{s}s")
    label = " ".join(parts)

    logger.info("Timer set: %d seconds (%s)", duration, label)

    from core import websocket_bridge as ws
    await ws.broadcast({
        "type":     "timer_start",
        "seconds":  duration,
        "label":    label,
    })

    return f"Timer set for {label}. I'll alert you when it's done."
