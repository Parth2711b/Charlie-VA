"""
llm/local_llm.py — Ollama API wrapper for local LLM inference.

Two model instances are used:
  - RouterLLM  (qwen2.5:1.5b)  — ultra-fast, single-token intent routing
  - AnswerLLM  (qwen2.5:3b)    — general conversation & search synthesis

Run:
  ollama pull qwen2.5:1.5b
  ollama pull qwen2.5:3b
"""

import logging
import httpx
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_ANSWER_MODEL, MAX_CONTEXT_TURNS

logger = logging.getLogger("Charlie.local_llm")

SYSTEM_PROMPT = (
    "You are Charlie, a personal AI voice assistant. "
    "Be concise — your responses will be spoken aloud. Keep them short and conversational. "
    "No markdown, no bullet points, no asterisks. "
    "If you don't know something, say so honestly."
)


class LocalLLM:
    """Wraps a single Ollama model. Instantiate once per model."""

    def __init__(self, model: str | None = None):
        self.base_url = OLLAMA_BASE_URL
        self.model = model or OLLAMA_MODEL
        self._check_connection()

    def _check_connection(self):
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            base = self.model.split(":")[0]
            if not any(base in m for m in models):
                logger.warning(
                    "Model '%s' not found in Ollama. Run: ollama pull %s",
                    self.model, self.model
                )
            else:
                logger.info("Ollama connected. Model: %s", self.model)
        except Exception as e:
            logger.error("Cannot connect to Ollama at %s: %s", self.base_url, e)
            logger.error("Make sure Ollama is running: https://ollama.ai")

    async def chat(self, user_input: str, context: list) -> str:
        """General conversation with context."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += context[-MAX_CONTEXT_TURNS * 2:]
        messages.append({"role": "user", "content": user_input})
        return await self._call(messages)

    async def answer_with_context(self, question: str, search_results: str, context: list) -> str:
        """Answer a question using web search results as context."""
        prompt = (
            f"Use the following search results to answer the question.\n\n"
            f"Search Results:\n{search_results}\n\n"
            f"Question: {question}\n\n"
            f"Give a concise spoken answer. No bullet points, no markdown."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
        return await self._call(messages)

    async def _call(self, messages: list) -> str:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model":    self.model,
                        "messages": messages,
                        "stream":   False,
                        "options":  {
                            "temperature": 0.7,
                            "num_predict": 200,   # keep responses short for TTS
                        }
                    }
                )
            data = response.json()
            return data["message"]["content"].strip()
        except Exception as e:
            import traceback
            logger.error("Ollama call failed (%s):\n%s", self.model, traceback.format_exc())
            return "I'm having trouble thinking right now. Please try again."


# ── Convenience singletons ─────────────────────────────────────────────────────

def get_router_llm() -> LocalLLM:
    """Tiny 1.5b model — only used for single-token routing decisions."""
    return LocalLLM(model=OLLAMA_MODEL)


def get_answer_llm() -> LocalLLM:
    """Larger 3b model — used for all actual answers to the user."""
    return LocalLLM(model=OLLAMA_ANSWER_MODEL)
