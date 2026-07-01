"""
core/intent_router.py
Classifies user intent and routes to the right handler.

Flow:
  1. Keyword match for action intents (fast, no LLM)
  2. Shutdown command check
  3. LLM decides: search needed or direct chat
  4. Offline fallback: local LLM only
"""

import logging
from config import is_online, CLOUD_LLM_ENABLED

logger = logging.getLogger("Charlie.router")

# ── Action keywords — fast pre-filter before LLM ──────────────────────────────
INTENT_KEYWORDS = {
    "whatsapp": ["whatsapp", "send message", "message to", "text to", "send a message"],
    "browser":  ["open", "go to", "visit", "browse", "youtube", "website", "url"],
    "vision":   ["look at", "what do you see", "camera", "screenshot", "screen"],
    "system":   ["volume up", "volume down", "mute", "open app", "clipboard"],
}

# ── Shutdown triggers ─────────────────────────────────────────────────────────
SHUTDOWN_WORDS = [
    "shut down", "shutdown", "goodbye charlie", "bye charlie",
    "stop charlie", "exit", "turn off", "power off"
]


def _keyword_match(text: str) -> str | None:
    """Fast keyword-based intent detection — runs before any LLM call."""
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return None


def _is_shutdown(text: str) -> bool:
    t = text.lower().strip()
    return any(w in t for w in SHUTDOWN_WORDS)


class IntentRouter:
    def __init__(self):
        from llm.local_llm import LocalLLM
        from llm.cloud_llm import CloudLLM
        from research.web_search import WebSearch
        from actions.whatsapp import WhatsAppAction
        from actions.browser import BrowserAction
        from actions.system import SystemAction
        from vision.screen_capture import ScreenCapture

        self.local_llm  = LocalLLM()
        self.cloud_llm  = CloudLLM() if CLOUD_LLM_ENABLED else None
        self.web_search = WebSearch()
        self.whatsapp   = WhatsAppAction()
        self.browser    = BrowserAction()
        self.system     = SystemAction()
        self.screen     = ScreenCapture()

    async def route(self, text: str, context: list) -> str:
        online = is_online()

        # ── 1. Shutdown check ──────────────────────────────────────────────────
        if _is_shutdown(text):
            return "__SHUTDOWN__"

        # ── 2. Action intents — keyword based ─────────────────────────────────
        intent = _keyword_match(text)

        try:
            if intent == "whatsapp":
                return await self.whatsapp.handle(text)

            elif intent == "browser":
                return await self.browser.handle(text)

            elif intent == "vision":
                return self.screen.capture_and_describe()

            elif intent == "system":
                return await self.system.handle(text)

        except Exception as e:
            logger.error("Action handler error for '%s': %s", intent, e)
            return "I ran into an issue with that. Please try again."

        # ── 3. LLM decides: search or chat ────────────────────────────────────
        if online:
            needs_search = await self._needs_search(text)

            if needs_search:
                logger.info("LLM decided: search needed")
                try:
                    results = await self.web_search.search(text)
                    llm = self.cloud_llm if self.cloud_llm else self.local_llm
                    return await llm.answer_with_context(text, results, context)
                except Exception as e:
                    logger.error("Search failed, falling back to LLM: %s", e)
                    return await self.local_llm.chat(text, context)

            else:
                logger.info("LLM decided: direct chat")
                return await self.local_llm.chat(text, context)

        # ── 4. Offline fallback ────────────────────────────────────────────────
        logger.info("Offline — using local LLM only")
        return await self.local_llm.chat(text, context)

    async def _needs_search(self, text: str) -> bool:
        """
        Ask local LLM if this query needs a web search.
        Returns True if search needed, False for direct chat.
        Fast call — small prompt, 1 token response.
        """
        prompt = (
            "Does the following query require searching the web for "
            "current events, live scores, recent news, factual data, "
            "prices, or anything that changes over time?\n\n"
            f"Query: \"{text}\"\n\n"
            "Reply with only YES or NO. Nothing else."
        )
        try:
            response = await self.local_llm._call([
                {"role": "user", "content": prompt}
            ])
            decision = response.strip().upper()
            logger.info("Search decision for '%s': %s", text[:40], decision)
            return "YES" in decision
        except Exception as e:
            logger.error("Search decision failed: %s — defaulting to search", e)
            return True  # safer to search than hallucinate