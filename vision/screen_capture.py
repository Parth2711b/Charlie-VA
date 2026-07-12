"""
vision/screen_capture.py - Screenshot + LLaVA vision understanding.
Uses Ollama's LLaVA model to semantically describe screen content.
No OCR - actual image understanding.

Changes:
  - Async: uses httpx.AsyncClient so we don't freeze Charlie's event loop.
  - In-memory: encodes screenshot to base64 directly via BytesIO (no temp file).
"""

import base64
import io
import logging

import httpx
from PIL import ImageGrab

logger = logging.getLogger("charlie.vision")

OLLAMA_URL   = "http://localhost:11434/api/generate"
VISION_MODEL = "llava:7b"


class ScreenCapture:
    async def capture_and_describe(self, task: str = "describe") -> str:
        """Take screenshot, send to LLaVA, return semantic description or translation."""
        try:
            # ── Capture screen ─────────────────────────────────────────────
            screenshot = ImageGrab.grab()

            # ── Encode to base64 IN MEMORY (no temp file!) ─────────────────
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # ── Dynamic Prompting ──────────────────────────────────────────
            if task == "translate":
                prompt = (
                    "Look at the text on this screen, especially anything that is highlighted or in focus. "
                    "Translate any foreign language text to English. Just provide the translation directly without introductory words."
                )
            elif task == "read":
                prompt = (
                    "Read the text on this screen aloud. Focus on the main content, article, or highlighted text. "
                    "Just provide the text directly."
                )
            else:
                prompt = (
                    "Describe what is on this screen in 2-3 short sentences. "
                    "Focus on what the user is doing or what app is open. "
                    "Be specific and concise - this will be spoken aloud."
                )

            # ── Send to LLaVA (ASYNC — doesn't freeze event loop) ──────────
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    OLLAMA_URL,
                    json={
                        "model":  VISION_MODEL,
                        "prompt": prompt,
                        "images": [img_b64],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 150},
                    },
                )

            result = response.json().get("response", "").strip()
            logger.info("Vision response for task '%s': %s", task, result)
            return result if result else "I couldn't read the screen properly."

        except Exception as e:
            logger.error("Screen capture error: %s", e)
            return "Couldn't capture or analyze the screen."