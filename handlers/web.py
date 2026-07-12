"""
handlers/web.py - Load any URL in the dashboard content panel.
"""

import re
import logging
import webbrowser

logger = logging.getLogger("Charlie.handler.web")

SITE_SHORTCUTS = {
    "youtube":   "https://www.youtube.com",
    "google":    "https://www.google.com",
    "gmail":     "https://mail.google.com",
    "github":    "https://www.github.com",
    "linkedin":  "https://www.linkedin.com",
    "maps":      "https://maps.google.com",
    "netflix":   "https://www.netflix.com",
    "spotify":   "https://open.spotify.com",
    "twitter":   "https://www.twitter.com",
    "reddit":    "https://www.reddit.com",
    "wikipedia": "https://www.wikipedia.org",
    "leetcode":  "https://www.leetcode.com",
    "chatgpt":   "https://chat.openai.com",
    "whatsapp":  "https://web.whatsapp.com",
}

# These sites block iframes - open in OS browser instead
IFRAME_BLOCKERS = {"netflix", "spotify", "gmail", "google", "youtube"}


def _extract_url(text: str) -> tuple[str, bool]:
    """Returns (url, use_iframe). use_iframe=False means open in OS browser."""
    text_lower = text.lower()

    for name, url in SITE_SHORTCUTS.items():
        if name in text_lower:
            return url, name not in IFRAME_BLOCKERS

    url_match = re.search(r"https?://\S+", text)
    if url_match:
        return url_match.group(0), True

    domain_match = re.search(r"(?:open|go to|visit|browse|load)\s+(\S+)", text_lower)
    if domain_match:
        site = domain_match.group(1).strip(".,!?")
        if "." not in site:
            site = f"{site}.com"
        url = f"https://{site}"
        return url, True

    return "", True


async def handle(text: str) -> str:
    url, use_iframe = _extract_url(text)

    if not url:
        return "Which site should I open? Try 'open github' or 'go to wikipedia'."

    if use_iframe:
        logger.info("Loading URL in dashboard: %s", url)
        from core import websocket_bridge as ws
        await ws.broadcast({"type": "load_url", "url": url, "mode": "url"})
        return f"Loading {url} in the dashboard."
    else:
        logger.info("Opening URL in OS browser (iframe blocked): %s", url)
        from core import websocket_bridge as ws
        await ws.broadcast({"type": "restricted_site", "url": url})
        webbrowser.open(url)
        return f"Opened {url} in your browser."
