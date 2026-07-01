"""
speech/stt.py — Speech-to-text using faster-whisper.
CPU-optimized: int8 compute type for speed on non-GPU machines.
"""

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop
import logging
import os
import tempfile
import wave

import pyaudio
from faster_whisper import WhisperModel

from config import WHISPER_LANGUAGE, WHISPER_MODEL

logger = logging.getLogger("charlie.stt")

# ── Audio Recording Settings ───────────────────────────────────────────────────
FORMAT         = pyaudio.paInt16
CHANNELS       = 1
RATE           = 16000
RECORD_SECONDS = 8      # max recording duration per utterance
CHUNK          = 1024

# ── Silence Detection ─────────────────────────────────────────────────────────
SILENCE_THRESHOLD  = 600   # RMS amplitude below this = silence
SILENCE_DURATION   = 1.5   # seconds of continuous silence to stop early
MIN_SPEECH_CHUNKS  = 5     # must detect this many loud chunks before accepting

# ── Transcription Quality ─────────────────────────────────────────────────────
MIN_TEXT_LENGTH    = 3     # discard transcriptions shorter than this
HALLUCINATION_PHRASES = [  # known Whisper hallucinations on silence
    "thank you", "thanks for watching", "bye", "goodbye",
    "subscribe", "see you", "you"
]


class STT:
    def __init__(self):
        logger.info("Loading Whisper model: %s (CPU, int8)", WHISPER_MODEL)
        # int8 = fastest on CPU, minor quality tradeoff — fine for voice commands
        self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        self.audio = pyaudio.PyAudio()
        logger.info("Whisper ready.")

    # ── Public Interface ───────────────────────────────────────────────────────

    def record_audio(self) -> str:
        """Record until silence or max duration. Returns path to temp .wav file."""
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        logger.info("Recording...")
        frames         = []
        silent_chunks  = 0
        speech_chunks  = 0
        max_silent     = int(RATE / CHUNK * SILENCE_DURATION)

        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data      = stream.read(CHUNK, exception_on_overflow=False)
            amplitude = audioop.rms(data, 2)
            frames.append(data)

            if amplitude < SILENCE_THRESHOLD:
                silent_chunks += 1
                if speech_chunks >= MIN_SPEECH_CHUNKS and silent_chunks >= max_silent:
                    logger.debug("Silence detected after speech — stopping early.")
                    break
            else:
                speech_chunks += 1
                silent_chunks  = 0

        stream.stop_stream()
        stream.close()

        return self._save_wav(frames)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe .wav file → cleaned text string. Returns empty string if invalid."""
        segments, info = self.model.transcribe(
            audio_path,
            language=WHISPER_LANGUAGE,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        logger.info("Raw transcription: '%s' (lang_prob=%.2f)", text, info.language_probability)

        if not self._is_valid(text):
            logger.info("Transcription rejected — likely silence or hallucination.")
            return ""

        return text

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _save_wav(self, frames: list) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        logger.debug("Audio saved to %s", tmp.name)
        return tmp.name

    def _is_valid(self, text: str) -> bool:
        if len(text) < MIN_TEXT_LENGTH:
            return False
        lower = text.lower().strip(" .,!")
        if lower in HALLUCINATION_PHRASES:
            return False
        return True

    def __del__(self):
        try:
            self.audio.terminate()
        except Exception:
            pass