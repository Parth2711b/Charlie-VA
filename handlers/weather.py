"""
handlers/weather.py - Fetch weather and display a card in the dashboard.
Uses wttr.in (free, no API key) for offline-friendly weather data.
"""

import logging
import httpx

logger = logging.getLogger("Charlie.handler.weather")


def _extract_city(text: str) -> str:
    """Extract city name from weather command."""
    import re
    text_lower = text.lower()
    patterns = [
        r"weather\s+(?:in|at|for|of)\s+([a-zA-Z\s]+?)(?:\s+today|$|\?)",
        r"(?:what(?:'s| is) the weather)\s+(?:in|at)\s+([a-zA-Z\s]+?)(?:\s+today|$|\?)",
        r"(?:temperature|temp)\s+(?:in|at|of)\s+([a-zA-Z\s]+?)(?:\s+today|$|\?)",
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            return m.group(1).strip().title()
    return "Pune"   # default city


async def handle(text: str) -> str:
    city = _extract_city(text)
    logger.info("Weather requested for: %s", city)

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            # wttr.in JSON API - no key needed
            r = await client.get(
                f"https://wttr.in/{city.replace(' ', '+')}?format=j1",
                headers={"User-Agent": "Charlie-VA/2.0"}
            )
            data = r.json()
            current = data["current_condition"][0]
            temp_c  = current["temp_C"]
            feels   = current["FeelsLikeC"]
            humid   = current["humidity"]
            desc    = current["weatherDesc"][0]["value"]
            wind    = current["windspeedKmph"]

            # Broadcast a weather card to the dashboard
            from core import websocket_bridge as ws
            await ws.emit({
                "type": "weather_card",
                "city": city,
                "temp_c": temp_c,
                "feels": feels,
                "humidity": humid,
                "description": desc,
                "wind": wind,
            })
            await ws.emit({"type": "focus_panel", "panel": "map"})

            return (
                f"In {city}: {desc}, {temp_c}°C, "
                f"feels like {feels}°C, humidity {humid}%, wind {wind} km/h."
            )

    except Exception as e:
        logger.error("Weather fetch failed: %s", e)
        return f"Couldn't get weather for {city} right now. Check your connection."
