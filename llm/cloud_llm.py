"""
llm/cloud_llm.py - Claude API for when internet is available and CLOUD_LLM_ENABLED=true.
Only used if explicitly enabled - local LLM is always the default.
"""

import logging
import anthropic
from config import ANTHROPIC_API_KEY, CLOUD_MODEL, MAX_CONTEXT_TURNS

logger = logging.getLogger("charlie.cloud_llm")

SYSTEM_PROMPT = """You are charlie, a personal AI voice assistant.
Be concise - your responses will be spoken aloud, so keep them short and conversational.
No markdown, no bullet points, no asterisks.
If you don't know something, say so honestly.
"""


class CloudLLM:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set. Cloud LLM will fail.")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model  = CLOUD_MODEL
        logger.info("Cloud LLM initialized. Model: %s", self.model)

    async def chat(self, user_input: str, context: list) -> str:
        messages = context[-MAX_CONTEXT_TURNS * 2:]
        messages.append({"role": "user", "content": user_input})
        return await self._call(messages)

    async def answer_with_context(self, question: str, search_results: str, context: list) -> str:
        prompt = (
            f"Use the following search results to answer the question.\n\n"
            f"Search Results:\n{search_results}\n\n"
            f"Question: {question}\n\n"
            f"Give a concise spoken answer. No bullet points."
        )
        messages = [{"role": "user", "content": prompt}]
        return await self._call(messages)

    async def _call(self, messages: list) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return "Cloud model unavailable. Switching to local."
