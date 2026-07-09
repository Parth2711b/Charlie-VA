"""
core/assistant.py — Main orchestrator.
Ties together: wake word → STT → intent → action → TTS
WebSocket bridge sends all events to dashboard in real time.
"""

import asyncio
import logging

from config import is_online

logger = logging.getLogger("Charlie.assistant")


class Assistant:
    def __init__(self):
        from speech.wake_word import WakeWordDetector
        from speech.stt import STT
        from speech.tts import TTS
        from core.intent_router import IntentRouter
        from core.memory import Memory
        from core import websocket_bridge as ws

        self.wake_word = WakeWordDetector()
        self.stt       = STT()
        self.tts       = TTS()
        self.router    = IntentRouter()
        self.memory    = Memory()
        self.ws        = ws

        # Register dashboard text input handler
        ws.set_text_input_handler(self._handle_text_input)

        logger.info("Assistant initialized. Online: %s", is_online())

    async def _speak(self, text: str):
        """Unified speaking method: route to dashboard if connected, else play locally."""
        if not text:
            return
            
        if len(self.ws._clients) > 0:
            logger.info("Streaming audio to dashboard clients...")
            b64_audio = self.tts.generate_audio_base64(text)
            if b64_audio:
                await self.ws.send_audio(b64_audio)
            else:
                self.tts.speak(text)
        else:
            self.tts.speak(text)

    async def run(self):
        """Main loop + WebSocket bridge running concurrently."""
        # Start WebSocket bridge as background task
        ws_task = asyncio.create_task(self.ws.start_server())
        await asyncio.sleep(0.5)  # let server start

        await self._speak("Charlie online. Ready.")
        await self.ws.send_status(stt="READY", llm="READY", mem="ACTIVE")

        logger.info("Entering main loop.")

        while True:
            try:
                # ── 1. Wait for wake word ──────────────────────────────────
                logger.info("Waiting for wake word... say 'charlie'")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.wake_word.wait_for_wake_word)

                # ── 2. Record + transcribe ────────────────────────────────
                await self._speak("Yes?")
                audio_path = await loop.run_in_executor(None, self.stt.record_audio)
                text = self.stt.transcribe(audio_path)

                if not text or len(text.strip()) < 2:
                    logger.info("Empty transcription, skipping.")
                    continue

                await self._process(text)

            except KeyboardInterrupt:
                logger.info("Shutting down.")
                await self._speak("Goodbye.")
                break
            except Exception as e:
                logger.error("Unhandled error in main loop: %s", e, exc_info=True)
                await self._speak("Something went wrong. Please try again.")

    async def _handle_text_input(self, text: str):
        """Handle text input from dashboard — same pipeline as voice."""
        await self._process(text)

    async def _process(self, text: str):
        """Core pipeline: text → intent → response → speak + dashboard update."""
        logger.info("Heard: %s", text)
        await self.ws.send_heard(text)

        # ── Load memory context ────────────────────────────────────────────
        context = self.memory.get_context()

        # ── Route intent → get response ───────────────────────────────────
        response = await self.router.route(text, context)

        # ── Shutdown command ───────────────────────────────────────────────────
        if response == "__SHUTDOWN__":
            await self._speak("Goodbye.")
            await self.ws.send_response("Shutting down.")
            import sys
            sys.exit(0)

        # ── Save to memory ────────────────────────────────────────────────
        self.memory.add_turn(user=text, assistant=response)

        # ── Send to dashboard ─────────────────────────────────────────────
        await self.ws.send_response(response)

        # ── Send updated memory facts ─────────────────────────────────────
        facts = list(self.memory.get_all_facts().values())
        if facts:
            await self.ws.send_memory(facts)

        # ── Speak response ────────────────────────────────────────────────────
        logger.info("Responding: %s", response)
        self.wake_word.pause()
        
        await self._speak(response)
            
        self.wake_word.resume()
        await asyncio.sleep(1.0)

        # ── Cooldown ──────────────────────────────────────────────────────────
        await asyncio.sleep(2.0)