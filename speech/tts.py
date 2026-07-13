"""
speech/tts.py - Text-to-speech via Piper TTS.
Piper is fast, offline, and light on CPU.
winsound used for playback on Windows - no media player window opens.

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
        self._last_audio_path = None  # tracks last played file for cleanup

        if not os.path.exists(self.voice_model):
            logger.warning(
                "Voice model not found at %s. "
                "Download from https://huggingface.co/rhasspy/piper-voices",
                self.voice_model
            )
        logger.info("TTS initialized. Voice: %s", TTS_VOICE)

    # ── Public Interface ───────────────────────────────────────────────────────

    def speak(self, text: str) -> float:
        """Convert text to speech and play it. Falls back to pyttsx3 on error. Returns duration."""
        if not text or not text.strip():
            return 0.0

        logger.debug("Speaking: %s", text[:80])

        try:
            return self._speak_piper(text)
        except Exception as e:
            logger.error("Piper failed... using fallback TTS")
            return self._speak_fallback(text)

    # ── Piper TTS ─────────────────────────────────────────────────────────────

    def _speak_piper(self, text: str) -> float:
        """Generate speech with Piper, play async, and return duration in seconds."""
        import wave

        # Clean up the PREVIOUS audio file (it's done playing by now).
        # We can't delete immediately after play because Windows SND_ASYNC
        # needs the file to exist while it plays. But by the NEXT speak call,
        # the old audio is definitely finished.
        if self._last_audio_path:
            try:
                os.unlink(self._last_audio_path)
            except OSError:
                pass  # file already gone, that's fine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        result = subprocess.run(
            [PIPER_BIN, "--model", self.voice_model, "--output_file", audio_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(f"Piper error: {result.stderr.decode()}")

        duration = 0.0
        try:
            with wave.open(audio_path, 'r') as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
        except Exception as e:
            logger.error("Failed to read wav duration: %s", e)

        self._play_wav(audio_path)
        self._last_audio_path = audio_path  # track it for cleanup next time
        return duration

    def generate_audio_base64(self, text: str) -> tuple[str, float]:
        """Generate speech and return as base64 encoded wav data and duration."""
        if not text or not text.strip():
            return "", 0.0
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        result = subprocess.run(
            [PIPER_BIN, "--model", self.voice_model, "--output_file", audio_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=300
        )

        if result.returncode != 0:
            logger.error("Piper failed in generate_audio_base64: %s", result.stderr.decode())
            return "", 0.0

        import base64
        import wave
        duration = 0.0
        try:
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                
            with open(audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error("Failed to read generated wav: %s", e)
            b64 = ""

        try:
            os.unlink(audio_path)
        except OSError:
            pass
        
        return b64, duration

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play_wav(self, path: str):
        """Play wav file async. Windows: winsound. Mac/Linux: system player in background."""
        if platform.system() == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif platform.system() == "Darwin":
            subprocess.Popen(["afplay", path])
        else:
            subprocess.Popen(["aplay", path])

    def stop(self):
        """Stop any currently playing audio."""
        if platform.system() == "Windows":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        else:
            subprocess.run(["pkill", "-f", "afplay|aplay"], capture_output=True)

    # ── Fallback TTS ──────────────────────────────────────────────────────────

    def _speak_fallback(self, text: str) -> float:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 175)
        # pyttsx3 blocks, so duration is just an estimate.
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return len(text) / 15.0  # rough estimate for wait time if we were async