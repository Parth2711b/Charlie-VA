"""
vision/ocr.py — Extract text from images using pytesseract.
Useful for reading documents, whiteboards, screenshots.

Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
Add to PATH or set TESSERACT_CMD in .env
"""

import logging
import os
import pytesseract
from PIL import Image

logger = logging.getLogger("Charlie.ocr")

# Windows Tesseract path — update if installed elsewhere
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class OCR:
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image file."""
        try:
            img  = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.error("OCR failed for %s: %s", image_path, e)
            return ""

    def extract_from_pil(self, pil_image) -> str:
        """Extract text directly from a PIL Image object."""
        try:
            return pytesseract.image_to_string(pil_image).strip()
        except Exception as e:
            logger.error("OCR from PIL failed: %s", e)
            return ""
