"""
core/assistant.py - Main orchestrator.
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
        self.memory    = Memory()
        self.router    = IntentRouter(self.memory)
        self.ws        = ws

        self.is_interviewing = False
        self.is_processing = False
        from handlers.interviewer_agent import InterviewerAgent
        self.interviewer = InterviewerAgent(self.memory)

        # Register dashboard text input handler
        ws.set_text_input_handler(self._handle_text_input)

        logger.info("Assistant initialized. Online: %s", is_online())

    async def _speak(self, text: str) -> float:
        """Unified speaking method: route to dashboard if connected, else play locally. Returns duration."""
        if not text:
            return 0.0
            
        if self.ws.has_clients():
            logger.info("Streaming audio to dashboard clients...")
            b64_audio = self.tts.generate_audio_base64(text)
            if b64_audio:
                await self.ws.send_audio(b64_audio)
                return len(text) / 15.0  # Estimate: ~15 chars per sec for TTS
            else:
                return self.tts.speak(text)
        else:
            return self.tts.speak(text)

    async def run(self):
        """Main loop + WebSocket bridge running concurrently."""
        # Start WebSocket bridge as background task
        ws_task = asyncio.create_task(self.ws.start_server())
        await asyncio.sleep(0.5)  # let server start

        await self._speak("Charlie online. Ready.")
        await self.ws.send_status(stt="READY", llm="READY", mem="ACTIVE")

        self.barge_in_triggered = False

        while True:
            try:
                # Get event loop ONCE at the top — avoids UnboundLocalError
                loop = asyncio.get_event_loop()

                # ── 1. Wait for wake word ──────────────────────────────────
                if not self.barge_in_triggered:
                    logger.info("Waiting for wake word... say 'charlie'")
                    await loop.run_in_executor(None, self.wake_word.wait_for_wake_word)
                else:
                    logger.info("Barge-in triggered, skipping wake word wait.")
                    self.barge_in_triggered = False

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
        """Handle text input from dashboard - same pipeline as voice."""
        await self._process(text)

    async def _process(self, text: str):
        """Core pipeline: text → intent → response → speak + dashboard update."""
        if self.is_processing:
            logger.warning("Already processing a query. Ignoring: %s", text)
            return
            
        self.is_processing = True
        try:
            await self._process_inner(text)
        finally:
            self.is_processing = False

    async def _process_inner(self, text: str):
        logger.info("Heard: %s", text)
        await self.ws.send_heard(text)

        # ── Load memory context ────────────────────────────────────────────
        context = self.memory.get_context()

        # ── RAG Context Injection (New) ───────────────────────────────────
        relevant_facts = self.memory.search_facts(text)
        if relevant_facts:
            facts_str = "\n- ".join(relevant_facts)
            # We secretly insert these facts into the context so the LLM reads them
            context.append({
                "role": "system", 
                "content": f"Relevant background facts about the user:\n- {facts_str}"
            })

        # ── Route intent → get response ───────────────────────────────────
        if self.is_interviewing:
            if any(w in text.lower() for w in ["stop interview", "end interview", "stop the interview", "quit interview"]):
                self.is_interviewing = False
                await self.ws.send_interview_mode(False)
                response = "Interview mode ended. Back to normal assistant mode."
                self.interviewer.reset()
            else:
                response = await self.interviewer.chat(text)
        else:
            response = await self.router.route(text, context)
            if response == "__START_INTERVIEW__":
                self.is_interviewing = True
                await self.ws.send_interview_mode(True)
                response = await self.interviewer.start_interview()

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
        
        duration = await self._speak(response)
        
        if duration > 0:
            logger.info("Listening for barge-in for %.1fs...", duration)
            loop = asyncio.get_event_loop()
            listen_task = loop.run_in_executor(None, self.wake_word.wait_for_wake_word)
            
            try:
                # Wait for the audio duration. If wake word finishes first, it throws no error.
                await asyncio.wait_for(listen_task, timeout=duration)
                logger.info("Barge-in detected!")
                self.tts.stop()
                await self.ws.send_stop_audio()
                self.barge_in_triggered = True
            except asyncio.TimeoutError:
                # Finished speaking naturally, no barge-in.
                # IMPORTANT: Cancel the orphaned listener so its audio stream closes!
                listen_task.cancel()
                pass

        await asyncio.sleep(0.5)