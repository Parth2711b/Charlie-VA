"""
speech/wake_word.py — Wake word detection via OpenWakeWord.
Blocks until wake word is detected, then returns.
"""

import logging
import numpy as np
import pyaudio
from openwakeword.model import Model
from config import WAKE_WORD_MODEL

logger = logging.getLogger("Charlie.wake_word")

CHUNK     = 1280
FORMAT    = pyaudio.paInt16
CHANNELS  = 1
RATE      = 16000
THRESHOLD = 0.5


class WakeWordDetector:
    def __init__(self):
        logger.info("Loading wake word model: %s", WAKE_WORD_MODEL)
        self.model   = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
        self.audio   = pyaudio.PyAudio()
        self._stream = None
        self._paused = False

    # ── Public Interface ───────────────────────────────────────────────────────

    def wait_for_wake_word(self):
        """Block until wake word is detected."""
        self._stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self._paused = False
        logger.debug("Listening for wake word...")

        try:
            while True:
                if self._paused:
                    import time
                    time.sleep(0.05)
                    continue

                audio_chunk  = self._stream.read(CHUNK, exception_on_overflow=False)
                audio_array  = np.frombuffer(audio_chunk, dtype=np.int16)
                prediction   = self.model.predict(audio_array)

                for model_name, score in prediction.items():
                    if score >= THRESHOLD:
                        logger.info("Wake word detected! (model=%s, score=%.2f)", model_name, score)
                        return
        finally:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def pause(self):
        """Pause mic during TTS to prevent re-trigger."""
        self._paused = True
        logger.debug("Wake word mic paused.")

    def resume(self):
        """Resume mic after TTS."""
        self._paused = False
        logger.debug("Wake word mic resumed.")

    def __del__(self):
        try:
            self.audio.terminate()
        except Exception:
            pass