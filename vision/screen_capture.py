"""
vision/screen_capture.py — Screenshot + describe using local LLM or OCR.
"""

import logging
import tempfile
import os
from PIL import ImageGrab
import pytesseract

logger = logging.getLogger("Charlie.screen_capture")


class ScreenCapture:
    def capture_and_describe(self) -> str:
        """Take screenshot, OCR it, return text summary."""
        try:
            screenshot = ImageGrab.grab()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                screenshot.save(tmp.name)
                path = tmp.name

            text = pytesseract.image_to_string(screenshot)
            os.unlink(path)

            if not text.strip():
                return "I took a screenshot but couldn't read any text on screen."

            # Truncate — screen can have a lot of text
            summary = text.strip()[:500]
            return f"Here's what I see on your screen: {summary}"

        except Exception as e:
            logger.error("Screen capture error: %s", e)
            return "Couldn't capture the screen."
