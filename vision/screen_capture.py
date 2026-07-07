"""
vision/screen_capture.py — Screenshot + LLaVA vision understanding.
Uses Ollama's LLaVA model to semantically describe screen content.
No OCR — actual image understanding.
"""

import base64
import logging
import tempfile
import os

import httpx
from PIL import ImageGrab

logger = logging.getLogger("charlie.vision")

OLLAMA_URL  = "http://localhost:11434/api/generate"
VISION_MODEL = "llava:7b"


class ScreenCapture:
    def capture_and_describe(self) -> str:
        """Take screenshot, send to LLaVA, return semantic description."""
        try:
            # ── Capture screen ─────────────────────────────────────────────
            screenshot = ImageGrab.grab()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                screenshot.save(tmp.name, format="PNG")
                tmp_path = tmp.name

            # ── Encode to base64 ───────────────────────────────────────────
            with open(tmp_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            os.unlink(tmp_path)

            # ── Send to LLaVA ──────────────────────────────────────────────
            response = httpx.post(
                OLLAMA_URL,
                json={
                    "model":  VISION_MODEL,
                    "prompt": (
                        "Describe what is on this screen in 2-3 short sentences. "
                        "Focus on what the user is doing or what app is open. "
                        "Be specific and concise — this will be spoken aloud."
                    ),
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 100},
                },
                timeout=60,
            )

            result = response.json().get("response", "").strip()
            logger.info("Vision response: %s", result)
            return result if result else "I couldn't understand what's on the screen."

        except Exception as e:
            logger.error("Screen capture error: %s", e)
            return "Couldn't capture or analyze the screen."