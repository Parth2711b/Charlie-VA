"""
research/web_search.py — Free web search via DuckDuckGo (no API key needed).
Falls back to empty string if offline or rate-limited.
"""

import logging
import asyncio
from ddgs import DDGS
from config import MAX_SEARCH_RESULTS, SEARCH_TIMEOUT

logger = logging.getLogger("Charlie.web_search")


class WebSearch:
    def __init__(self):
        self.ddgs = DDGS()

    async def search(self, query: str) -> str:
        """Run a web search and return formatted results string for LLM context."""
        logger.info("Searching: %s", query)

        try:
            # DDGS is sync — run in executor to not block async loop
            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_search, query),
                timeout=SEARCH_TIMEOUT
            )

            if not results:
                logger.warning("No results for: %s", query)
                return "No search results found."

            formatted = self._format_results(results)
            logger.info("Got %d results for '%s'", len(results), query)
            return formatted

        except asyncio.TimeoutError:
            logger.error("Search timed out for: %s", query)
            return "Search timed out."
        except Exception as e:
            logger.error("Search error: %s", e)
            return "Search failed."

    def _sync_search(self, query: str) -> list:
        return list(self.ddgs.text(query, max_results=MAX_SEARCH_RESULTS))

    def _format_results(self, results: list) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body  = r.get("body", "")
            href  = r.get("href", "")
            lines.append(f"[{i}] {title}\n{body}\nSource: {href}")
        return "\n\n".join(lines)
