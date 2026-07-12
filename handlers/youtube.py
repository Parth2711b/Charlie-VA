"""
handlers/youtube.py - Play YouTube videos inside the dashboard panel.
Searches YouTube server-side (no API key needed) to get a real video ID,
then sends an embeddable URL to the dashboard via WebSocket.
"""

import re
import logging
from urllib.parse import quote_plus

logger = logging.getLogger("Charlie.handler.youtube")


def _extract_query(text: str) -> str:
    """Pull the search query out of 'play X on youtube' style commands."""
    text_lower = text.lower()
    patterns = [
        r"(?:play|search|find|show)\s+(.+?)\s+(?:on\s+)?youtube",
        r"youtube\s+(?:play|search|find)\s+(.+)",
        r"(?:play|search)\s+(.+?)(?:\s+song|\s+video|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            return m.group(1).strip()
            
    # Fallback: if we matched the youtube intent but no verb, just strip "youtube" and "on"
    fallback = text_lower.replace("youtube", "").replace("on ", "").strip(" .,!?")
    if len(fallback) > 2:
        return fallback
    return ""


async def _get_first_video_id(query: str) -> str:
    """
    Search YouTube's website server-side and extract the first video ID.
    No API key required - parses the videoId JSON embedded in the HTML.
    Returns empty string on failure.
    """
    try:
        import httpx
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return ""
            # YouTube embeds all video data as JSON in the page source
            matches = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
            if matches:
                return matches[0]
    except Exception as e:
        logger.warning("YouTube search scrape failed: %s", e)
    return ""


async def handle(text: str) -> str:
    query = _extract_query(text)

    from core import websocket_bridge as ws

    if not query:
        import webbrowser
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube in your browser."

    logger.info("Searching YouTube server-side for: %s", query)
    video_id = await _get_first_video_id(query)

    if video_id:
        embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
        logger.info("YouTube embed: %s -> %s", query, video_id)
        await ws.broadcast({"type": "load_url", "url": embed_url, "mode": "url"})
        return f"Playing '{query}' on YouTube in the dashboard."
    else:
        import webbrowser
        webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
        logger.warning("YouTube scrape failed for '%s' - opened in browser", query)
        return f"Couldn't embed '{query}' - opened YouTube search in your browser instead."
