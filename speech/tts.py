"""
speech/tts.py - Text-to-speech via Microsoft Edge TTS.
Ultra-realistic cloud neural voices.
"""

import logging
import os
import platform
import subprocess
import tempfile
import asyncio
import edge_tts

from config import TTS_VOICE

logger = logging.getLogger("charlie.tts")

class TTS:
    def __init__(self):
        self._last_audio_path = None
        logger.info("Edge TTS initialized. Voice: %s", TTS_VOICE)

    async def speak(self, text: str) -> float:
        """Convert text to speech and play it locally."""
        if not text or not text.strip():
            return 0.0

        logger.debug("Speaking (Edge): %s", text[:80])
        try:
            return await self._speak_edge(text)
        except Exception as e:
            logger.error("Edge TTS failed: %s", e)
            return 0.0

    async def _speak_edge(self, text: str) -> float:
        if self._last_audio_path:
            try:
                os.unlink(self._last_audio_path)
            except OSError:
                pass

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name

        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(audio_path)

        self._play_audio(audio_path)
        self._last_audio_path = audio_path
        return len(text) / 15.0

    async def generate_audio_base64(self, text: str) -> tuple[str, float]:
        """Generate speech and return as base64 encoded MP3 data and duration."""
        if not text or not text.strip():
            return "", 0.0
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name

        try:
            communicate = edge_tts.Communicate(text, TTS_VOICE)
            await communicate.save(audio_path)
        except Exception as e:
            logger.error("Edge TTS generation failed: %s", e)
            return "", 0.0

        import base64
        try:
            with open(audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error("Failed to read generated edge audio: %s", e)
            b64 = ""

        try:
            os.unlink(audio_path)
        except OSError:
            pass
        
        estimated_duration = len(text) / 15.0
        return b64, estimated_duration

    def _play_audio(self, path: str):
        """Play audio file async. Windows uses winmm.dll to play MP3 silently."""
        if platform.system() == "Windows":
            import ctypes
            winmm = ctypes.windll.winmm
            # Close previous if exists, then open and play new silently
            winmm.mciSendStringA(b'close media', None, 0, 0)
            cmd = f'open "{path}" alias media'.encode('utf-8')
            winmm.mciSendStringA(cmd, None, 0, 0)
            winmm.mciSendStringA(b'play media', None, 0, 0)
        elif platform.system() == "Darwin":
            subprocess.Popen(["afplay", path])
        else:
            subprocess.Popen(["aplay", path])

    def stop(self):
        """Stop any currently playing audio."""
        if platform.system() == "Windows":
            import ctypes
            ctypes.windll.winmm.mciSendStringA(b'close media', None, 0, 0)
        else:
            subprocess.run(["pkill", "-f", "afplay|aplay"], capture_output=True)

    # ── Fallback TTS ──────────────────────────────────────────────────────────

    def _speak_fallback(self, text: str) -> float:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices and isinstance(voices, list):
            engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 175)
        # pyttsx3 blocks, so duration is just an estimate.
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return len(text) / 15.0  # rough estimate for wait time if we were async