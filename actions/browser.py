"""
actions/browser.py — Open websites and perform basic browser actions via Playwright.

Install: pip install playwright && python -m playwright install chromium
"""

import logging
import re
import asyncio
from playwright.async_api import async_playwright
from config import BROWSER_HEADLESS

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
        url = _extract_url(text)

        if not url:
            return "I couldn't figure out which site to open. Try saying 'open youtube' or 'go to github'."

        logger.info("Opening browser: %s", url)

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=BROWSER_HEADLESS)
                context = await browser.new_context()
                page    = await context.new_page()
                await page.goto(url, timeout=15000)
                await asyncio.sleep(2)  # let page load visually
                # Don't close — user wants to interact with the page
                # Browser stays open; just detach

            return f"Opened {url}."
        except Exception as e:
            logger.error("Browser error: %s", e)
            return f"Couldn't open {url}. Make sure Playwright is installed: python -m playwright install chromium"
