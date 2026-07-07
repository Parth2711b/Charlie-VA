"""
speech/tts.py — Text-to-speech via Piper TTS.
Piper is fast, offline, and light on CPU.
winsound used for playback on Windows — no media player window opens.

Setup:
  - Piper binary: https://github.com/rhasspy/piper/releases
  - Voice model:  https://huggingface.co/rhasspy/piper-voices
  - Place .onnx + .onnx.json in models/tts/
  - Set PIPER_BIN in .env to full path of piper.exe
"""

import logging
import os
import platform
import subprocess
import tempfile

from config import TTS_VOICE

logger = logging.getLogger("charlie.tts")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "tts")
PIPER_BIN  = os.getenv("PIPER_BIN", "piper")


class TTS:
    def __init__(self):
        self.voice_model = os.path.join(MODELS_DIR, f"{TTS_VOICE}.onnx")

        if not os.path.exists(self.voice_model):
            logger.warning(
                "Voice model not found at %s. "
                "Download from https://huggingface.co/rhasspy/piper-voices",
                self.voice_model
            )
        logger.info("TTS initialized. Voice: %s", TTS_VOICE)

    # ── Public Interface ───────────────────────────────────────────────────────

    def speak(self, text: str):
        """Convert text to speech and play it. Falls back to pyttsx3 on error."""
        if not text or not text.strip():
            return

        logger.debug("Speaking: %s", text[:80])

        try:
            self._speak_piper(text)
        except Exception as e:
            logger.error("Piper failed... using fallback TTS")
            self._speak_fallback(text)

    # ── Piper TTS ─────────────────────────────────────────────────────────────

    def _speak_piper(self, text: str):
        """Generate speech with Piper and play via winsound (no browser window)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        result = subprocess.run(
            [PIPER_BIN, "--model", self.voice_model, "--output_file", audio_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"Piper error: {result.stderr.decode()}")

        self._play_wav(audio_path)

        try:
            os.unlink(audio_path)
        except OSError:
            pass

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play_wav(self, path: str):
        """Play wav file. Windows: winsound (no popup). Mac/Linux: system player."""
        if platform.system() == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        elif platform.system() == "Darwin":
            subprocess.run(["afplay", path], check=True)
        else:
            subprocess.run(["aplay", path], check=True)

    # ── Fallback TTS ──────────────────────────────────────────────────────────

    def _speak_fallback(self, text: str):
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        # Index 0 = male (David), Index 1 = female (Zira) on Windows
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
        engine.stop()