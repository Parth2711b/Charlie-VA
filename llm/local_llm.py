"""
llm/local_llm.py - Ollama API wrapper for local LLM inference.

Two model instances are used:
  - RouterLLM  (qwen2.5:1.5b)  - ultra-fast, single-token intent routing
  - AnswerLLM  (qwen2.5:3b)    - general conversation & search synthesis

Performance optimizations:
  - Persistent httpx.AsyncClient (reuses TCP connection, no handshake per call)
  - keep_alive: "30m" tells Ollama to keep model loaded in RAM
  - Actual singletons via module-level caching
  - Single connection check at startup

Run:
  ollama pull qwen2.5:1.5b
  ollama pull qwen2.5:3b
"""

import logging
import httpx
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_ANSWER_MODEL, MAX_CONTEXT_TURNS

logger = logging.getLogger("Charlie.local_llm")

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

# ── One-time connection check (shared across all instances) ────────────────────
_connection_verified = False

def _verify_ollama_once():
    """Check Ollama is reachable — called once at startup, not per-instance."""
    global _connection_verified
    if _connection_verified:
        return
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        logger.info("Ollama connected. Available models: %s", ", ".join(models))
        _connection_verified = True
    except Exception as e:
        logger.error("Cannot connect to Ollama at %s: %s", OLLAMA_BASE_URL, e)
        logger.error("Make sure Ollama is running: https://ollama.ai")


class LocalLLM:
    """Wraps a single Ollama model. Uses a persistent HTTP client for speed."""

    def __init__(self, model: str | None = None):
        self.base_url = OLLAMA_BASE_URL
        self.model = model or OLLAMA_MODEL

        # Persistent client — reuses the same TCP connection across all calls.
        # Limits added to prevent connection pool exhaustion during concurrent bursts.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120,   # generous timeout for slow CPU inference
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
        )

        _verify_ollama_once()
        logger.info("LLM ready: %s", self.model)

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
            response = await self._client.post(
                "/api/chat",
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   False,
                    "keep_alive": "30m",  # Keep model loaded in RAM for 30 min!
                    "options":  {
                        "temperature": 0.7,
                        "num_predict": 150,  # shorter = faster TTS + faster generation
                    }
                }
            )
            data = response.json()
            return data["message"]["content"].strip()
        except Exception as e:
            import traceback
            logger.error("Ollama call failed (%s):\n%s", self.model, traceback.format_exc())
            return "I'm having trouble thinking right now. Please try again."

    async def close(self):
        """Clean up the persistent client."""
        await self._client.aclose()


# ── True Singletons ───────────────────────────────────────────────────────────
# Before: get_router_llm() created a NEW LocalLLM every time it was called.
# Now: creates once, returns the cached instance forever.

_answer_llm = None


def get_answer_llm() -> LocalLLM:
    """Larger 3b model - used for all actual answers to the user."""
    global _answer_llm
    if _answer_llm is None:
        _answer_llm = LocalLLM(model=OLLAMA_ANSWER_MODEL)
    return _answer_llm
