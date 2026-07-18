"""
llm/groq_llm.py - Groq API wrapper for ultra-fast LLM inference.
"""

import logging
from groq import AsyncGroq
from config import GROQ_API_KEY, GROQ_MODEL, MAX_CONTEXT_TURNS

logger = logging.getLogger("Charlie.groq_llm")

SYSTEM_PROMPT = (
    "You are Charlie, a highly advanced, exceptionally intelligent, and slightly sarcastic AI assistant modeled after Jarvis from Iron Man. "
    "You were created by Parth Bansal, who is speaking to you now. Address him politely as Parth, or occasionally 'Sir'. "
    "If asked 'Who am I?', you MUST respond that they are Parth. "
    "Your tone should be highly polite, witty, dry, and relentlessly efficient. "
    "You are capable of seeing the screen, reading the camera, searching the web, checking live weather/flights, "
    "setting timers, taking notes, playing music via Spotify, and conducting mock interviews. "
    "IMPORTANT: If the user asks you to open a website, you CAN do so by simply outputting the URL in your response. The dashboard will automatically intercept the URL and open it for the user. "
    "CRITICAL ROUTING RULE: If the user asks for directions, a path, or a route between two places, you MUST output a Google Maps direction URL in the format https://www.google.com/maps/dir/<Origin>/<Destination> so the dashboard can draw it on the map. "
    "CRITICAL LENGTH RULE: Your responses are spoken aloud. "
    "Be as brief and concise as humanly possible for ALL answers. If a question can be answered in a few words, do so. "
    "ONLY give long, detailed answers if the user explicitly asks for an explanation, an essay, or a detailed story."
    "No markdown, no bullet points, no asterisks. "
    "If you don't know something, admit it with dry wit."
)


class GroqLLM:
    """Wraps a single Groq model."""

    def __init__(self, model: str | None = None):
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY is not set. Groq will fail.")
        
        self.model = model or GROQ_MODEL
        self._client = AsyncGroq(api_key=GROQ_API_KEY)
        logger.info("Groq LLM ready: %s", self.model)

    async def chat(self, user_input: str, context: list, relevant_facts: list[str] | None = None) -> str:
        """General conversation with context."""
        sys_prompt = SYSTEM_PROMPT
        if relevant_facts:
            facts_str = "\n- ".join(relevant_facts)
            sys_prompt += f"\n\nHere are some relevant memory facts you have learned about the user:\n- {facts_str}"
            
        messages = [{"role": "system", "content": sys_prompt}]
        messages += context[-MAX_CONTEXT_TURNS * 2:]
        messages.append({"role": "user", "content": user_input})
        return await self._call(messages)

    async def answer_with_context(self, question: str, search_results: str, context: list, relevant_facts: list[str] | None = None) -> str:
        """Answer a question using web search results as context."""
        sys_prompt = SYSTEM_PROMPT
        if relevant_facts:
            facts_str = "\n- ".join(relevant_facts)
            sys_prompt += f"\n\nHere are some relevant memory facts you have learned about the user:\n- {facts_str}"
            
        prompt = (
            f"Use the following search results to answer the question.\n\n"
            f"Search Results:\n{search_results}\n\n"
            f"Question: {question}\n\n"
            f"Give a concise spoken answer. No bullet points, no markdown."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": prompt}
        ]
        return await self._call(messages)

    async def _call(self, messages: list) -> str:
        try:
            chat_completion = await self._client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=300,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            import traceback
            logger.error("Groq call failed (%s):\n%s", self.model, traceback.format_exc())
            return "I'm having trouble thinking right now. Please try again."

    async def close(self):
        """Clean up the persistent client."""
        await self._client.close()


# ── True Singletons ───────────────────────────────────────────────────────────

_answer_llm = None


def get_answer_llm() -> GroqLLM:
    """Larger model - used for all actual answers to the user."""
    global _answer_llm
    if _answer_llm is None:
        _answer_llm = GroqLLM(model=GROQ_MODEL)
    return _answer_llm
