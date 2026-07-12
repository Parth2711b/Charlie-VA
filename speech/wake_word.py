"""
speech/wake_word.py - Keyword-based wake word detection via Vosk.
Continuously listens for the keyword "charlie" in a small audio window.
No account, no binary, no cloud - fully offline.
"""

import json
import logging
import time

import numpy as np
import pyaudio
# pyrefly: ignore [missing-import]
from vosk import KaldiRecognizer, Model, SetLogLevel

logger = logging.getLogger("Charlie.wake_word")

# ── Audio settings ─────────────────────────────────────────────────────────────
CHUNK    = 4000   # ~250ms chunks @ 16kHz
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000

# ── Wake keywords - any of these trigger activation ───────────────────────────
WAKE_KEYWORDS = ["charlie", "charley", "charly"]

# ── Vosk model path ───────────────────────────────────────────────────────────
import os
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "vosk"
)


class WakeWordDetector:
    def __init__(self):
        SetLogLevel(-1)  # suppress Vosk verbose output
        logger.info("Loading Vosk model from %s", MODEL_PATH)
        self.vosk_model = Model(MODEL_PATH)
        self.audio      = pyaudio.PyAudio()
        self._stream    = None
        self._paused    = False
        logger.info("Wake word detector ready. Keywords: %s", WAKE_KEYWORDS)

    # ── Public Interface ───────────────────────────────────────────────────────

    def wait_for_wake_word(self):
        """Block until a wake keyword is detected in the audio stream."""
        rec = KaldiRecognizer(self.vosk_model, RATE)
        rec.SetWords(True)

        self._stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self._paused = False
        logger.debug("Listening for wake keywords: %s", WAKE_KEYWORDS)

        try:
            while True:
                if self._paused:
                    time.sleep(0.05)
                    continue

                data = self._stream.read(CHUNK, exception_on_overflow=False)

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").lower()
                else:
                    result = json.loads(rec.PartialResult())
                    text   = result.get("partial", "").lower()

                if text and any(kw in text for kw in WAKE_KEYWORDS):
                    logger.info("Wake keyword detected: '%s'", text.strip())
                    return

        finally:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None

    def pause(self):
        """Pause detection during TTS to prevent self-trigger."""
        self._paused = True
        logger.debug("Wake word detection paused.")

    def resume(self):
        """Resume detection after TTS."""
        self._paused = False
        logger.debug("Wake word detection resumed.")

    def __del__(self):
        try:
            self.audio.terminate()
        except Exception:
            pass