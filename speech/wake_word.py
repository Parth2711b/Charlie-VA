"""
speech/wake_word.py - Keyword-based wake word detection via Vosk.
Continuously listens for the keyword "charlie" in a small audio window.
No account, no binary, no cloud - fully offline.
"""

import json
import logging
import time

import numpy as np
import pyaudio # type: ignore
# pyrefly: ignore [missing-import]
from vosk import KaldiRecognizer, Model, SetLogLevel

logger = logging.getLogger("Charlie.wake_word")

# ── Audio settings ─────────────────────────────────────────────────────────────
CHUNK    = 4000   # ~250ms chunks @ 16kHz
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000

# ── Wake keywords - any of these trigger activation ───────────────────────────
# We include phonetically similar words because offline STT often mishears names.
WAKE_KEYWORDS = ["charlie", "charley", "charly", "shirley", "harley", "barley", "trolley", "olly"]

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

    def wait_for_wake_word(self) -> bool:
        if getattr(self, "_is_listening", False):
            self._stop_requested = False
            while getattr(self, "_is_listening", False) and not self._stop_requested:
                time.sleep(0.05)
            return getattr(self, "_detected", False)

        self._is_listening = True
        self._stop_requested = False
        self._paused = False
        self._detected = False

        rec = KaldiRecognizer(self.vosk_model, RATE)
        rec.SetWords(True)

        try:
            self._stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            logger.debug("Listening for wake keywords: %s", WAKE_KEYWORDS)

            while not self._stop_requested:
                if self._paused:
                    time.sleep(0.05)
                    continue

                if not self._stream:
                    time.sleep(0.05)
                    continue
                try:
                    data = self._stream.read(CHUNK, exception_on_overflow=False)
                except Exception as e:
                    break

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").lower()
                else:
                    result = json.loads(rec.PartialResult())
                    text   = result.get("partial", "").lower()

                if text and any(kw in text for kw in WAKE_KEYWORDS):
                    logger.info("Wake keyword detected: '%s'", text.strip())
                    self._detected = True
                    return True

            return False

        finally:
            self._paused = False
            self._stop_requested = False
            self._is_listening = False
            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def stop_listening(self):
        self._stop_requested = True

    def pause(self):
        self._paused = True
        logger.debug("Wake word detection paused.")

    def resume(self):
        self._paused = False
        logger.debug("Wake word detection resumed.")

    def __del__(self):
        try:
            self.audio.terminate()
        except Exception:
            pass