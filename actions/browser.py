"""
actions/browser.py - Open websites and perform basic browser actions via Playwright.

Install: pip install playwright && python -m playwright install chromium
"""

import logging
import re
import webbrowser

logger = logging.getLogger("Charlie.browser")

# Common shorthand site mappings
SITE_SHORTCUTS = {
    "youtube":   "https://www.youtube.com",
    "google":    "https://www.google.com",
    "gmail":     "https://mail.google.com",
    "github":    "https://www.github.com",
    "linkedin":  "https://www.linkedin.com",
    "whatsapp":  "https://web.whatsapp.com",
    "maps":      "https://maps.google.com",
    "netflix":   "https://www.netflix.com",
    "spotify":   "https://open.spotify.com",
    "twitter":   "https://www.twitter.com",
    "reddit":    "https://www.reddit.com",
    "wikipedia": "https://www.wikipedia.org",
    "leetcode":  "https://www.leetcode.com",
}


def _extract_url(text: str) -> str:
    """Extract or build URL from spoken command."""
    text_lower = text.lower()

    # Check shortcuts first
    for name, url in SITE_SHORTCUTS.items():
        if name in text_lower:
            return url

    # Check if URL already in text
    url_match = re.search(r"https?://\S+", text)
    if url_match:
        return url_match.group(0)

    # Extract domain-like word and build URL
    domain_match = re.search(r"(?:open|go to|visit|browse)\s+(\S+)", text_lower)
    if domain_match:
        site = domain_match.group(1).strip(".,!?")
        if "." not in site:
            site = f"{site}.com"
        return f"https://{site}"

    return ""


class BrowserAction:
    async def handle(self, text: str) -> str:
        text_lower = text.lower()
        
        # YouTube search - send to dashboard instead of opening externally
        yt_match = re.search(
            r"(?:play|search|find)\s+(.+?)\s+(?:on\s+)?youtube", text_lower
        )
        if yt_match:
            query = yt_match.group(1).strip()
            url = f"https://www.youtube-nocookie.com/embed?listType=search&list={query.replace(' ', '+')}"
            try:
                from core import websocket_bridge as ws
                await ws.broadcast({"type": "load_url", "url": url, "mode": "yt"})
                await ws.broadcast({"type": "focus_panel", "panel": "map"})
                return f"Playing {query} on YouTube in dashboard."
            except Exception:
                pass  # fallback to external browser below
                
        url = _extract_url(text)

        if not url:
            return "I couldn't figure out which site to open. Try saying 'open youtube' or 'go to github'."

        logger.info("Opening browser: %s", url)

        try:
            webbrowser.open(url)
            return f"Opened {url}."
        except Exception as e:
            logger.error("Browser error: %s", e)
            return f"Couldn't open {url}."
