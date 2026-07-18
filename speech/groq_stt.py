"""
speech/groq_stt.py - Speech-to-text using Groq's whisper API.
Ultra-fast transcription in the cloud.
"""

import logging
import os
import tempfile
import wave
try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop

import pyaudio
from groq import Groq

from config import GROQ_API_KEY

logger = logging.getLogger("charlie.groq_stt")

# ── Audio Recording Settings ───────────────────────────────────────────────────
FORMAT         = pyaudio.paInt16
CHANNELS       = 1
RATE           = 16000
RECORD_SECONDS = 8      # max recording duration per utterance
CHUNK          = 1024

# ── Silence Detection ─────────────────────────────────────────────────────────
SILENCE_THRESHOLD  = 600   # RMS amplitude below this = silence
SILENCE_DURATION   = 2.5   # seconds of continuous silence to stop early
MIN_SPEECH_CHUNKS  = 5     # must detect this many loud chunks before accepting
MIN_TEXT_LENGTH    = 3     # discard transcriptions shorter than this


class GroqSTT:
    def __init__(self):
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY is not set. Groq STT will fail.")
            
        self.client = Groq(api_key=GROQ_API_KEY)
        self.audio = pyaudio.PyAudio()
        logger.info("Groq STT ready.")

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
                    logger.debug("Silence detected after speech - stopping early.")
                    break
            else:
                speech_chunks += 1
                silent_chunks  = 0

        stream.stop_stream()
        stream.close()

        return self._save_wav(frames)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe .wav file → cleaned text string using Groq API."""
        try:
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                  file=(audio_path, file.read()),
                  model="whisper-large-v3",
                  prompt="Specify context or spelling",
                  response_format="json",
                  language="en",
                  temperature=0.0
                )
            text = transcription.text.strip()
            
            logger.info("Groq raw transcription: '%s'", text)
            
            if len(text) < MIN_TEXT_LENGTH:
                return ""
            
            return text
        except Exception as e:
            logger.error("Groq STT transcription failed: %s", e)
            return ""

    def _save_wav(self, frames: list) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        logger.debug("Audio saved to %s", tmp.name)
        return tmp.name

    def __del__(self):
        try:
            self.audio.terminate()
        except Exception:
            pass
