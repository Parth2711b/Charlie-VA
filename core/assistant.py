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
        from config import USE_GROQ_STT
        if USE_GROQ_STT:
            from speech.groq_stt import GroqSTT as STT
        else:
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
        
        self.is_processing = False
        self.barge_in_triggered = False
        self.expecting_reply = False
        self.is_interviewing = {}
        from handlers.interviewer_agent import InterviewerAgent
        self.interviewer = InterviewerAgent(self.memory)

        # Register dashboard input handlers
        ws.set_text_input_handler(self._handle_text_input)
        ws.set_audio_input_handler(self._handle_audio_input)
        ws.set_barge_in_handler(self._handle_barge_in)
        
        self.barge_in_triggered = False

        logger.info("Assistant initialized. Online: %s", is_online())

    async def _speak(self, text: str) -> float:
        """Unified speaking method: route to dashboard if connected, else play locally. Returns duration."""
        if not text:
            return 0.0
            
        user_id = self.ws.current_user_id.get() or 1
        
        if self.ws.has_clients():
            logger.info("Generating Edge TTS audio for dashboard...")
            
            if self.barge_in_triggered:
                logger.info("TTS aborted due to barge-in.")
                self.barge_in_triggered = False
                return 0.0
                
            # Direct async generation!
            b64_audio, duration = await self.tts.generate_audio_base64(text)
            if b64_audio:
                await self.ws.send_audio(b64_audio, target_user_id=user_id)
                
            self.barge_in_triggered = False
            return duration
        else:
            return await self.tts.speak(text)

    async def run(self):
        """Main entry point. Starts WebSockets and background tasks. The main loop is fully event-driven now."""
        # Start WebSocket bridge as background task
        ws_task = asyncio.create_task(self.ws.start_server())
        
        # Start background reminder loop
        reminder_task = asyncio.create_task(self._reminder_loop())
        
        # Start background wake word loop
        wakeword_task = asyncio.create_task(self._wake_word_loop())
        
        await asyncio.sleep(0.5)  # let server start

        # Do NOT speak locally on boot in a multi-tenant environment.
        # Wait for users to connect and authenticate.

        # Keep the process alive while event-driven handlers do the work
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        except Exception as e:
            logger.error("Unhandled error: %s", e)

    async def _wake_word_loop(self):
        """Background loop to detect the 'charlie' wake word using local microphone."""
        while True:
            try:
                loop = asyncio.get_event_loop()
                
                if self.expecting_reply:
                    detected = True
                    self.expecting_reply = False
                    # Give a tiny delay for audio to finish playing if any
                    await asyncio.sleep(0.5)
                else:
                    detected = await loop.run_in_executor(None, self.wake_word.wait_for_wake_word)
                
                if detected:
                    logger.info("Wake word detected locally!")
                    self.barge_in_triggered = True
                    # Abort any playing audio on the dashboard
                    await self.ws.send_stop_audio(target_user_id=1)
                    # Acknowledge only if it was an actual wake word detection
                    if detected is not True:
                        await self._speak("Yes?")
                    
                    # Tell dashboard we are actively recording locally
                    await self.ws.send_listening(True, target_user_id=1)
                    
                    # Record locally via PyAudio
                    audio_path = await loop.run_in_executor(None, self.stt.record_audio)
                    
                    # Done recording
                    await self.ws.send_listening(False, target_user_id=1)
                    
                    text = await loop.run_in_executor(None, self.stt.transcribe, audio_path)
                    
                    if text and len(text.strip()) >= 2:
                        # process the text in the background so we don't block the wake word loop forever
                        asyncio.create_task(self._process(text))
            except Exception as e:
                logger.error("Wake word loop error: %s", e)
                await asyncio.sleep(2)

    async def _reminder_loop(self):
        """Background loop to check for due scheduled events."""
        from core.schedule_db import schedule_db
        while True:
            try:
                due_events = schedule_db.get_due_events()
                for ev in due_events:
                    user_id = ev['user_id']
                    logger.info("Reminder triggered for user %s: %s", user_id, ev['event_text'])
                    
                    msg = f"Reminder: You scheduled a task for now: {ev['event_text']}. Have you done it?"
                    
                    # Generate audio and send to specific user
                    b64_audio, _ = await self.tts.generate_audio_base64(msg)
                    if b64_audio:
                        await self.ws.emit({"type": "audio", "data": b64_audio}, target_user_id=user_id)
                        
                    await self.ws.emit({"type": "response", "text": msg}, target_user_id=user_id)
                    
                    # Mark as done so it doesn't repeat
                    schedule_db.mark_done(ev['id'])
                    # Notify dashboard
                    await self.ws.emit({"type": "schedule_update"}, target_user_id=user_id)
            except Exception as e:
                logger.error("Reminder loop error: %s", e)
            await asyncio.sleep(60)

    async def _handle_text_input(self, text: str):
        """Handle text input from dashboard."""
        self.barge_in_triggered = True
        await self._process(text)

    def _handle_barge_in(self):
        """Called when user starts recording or types."""
        self.barge_in_triggered = True

    async def _handle_audio_input(self, audio_bytes: bytes):
        """Handle binary audio uploaded from the dashboard via WebSocket."""
        import tempfile
        import os
        
        # Save bytes to a temp webm file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
            
        try:
            logger.info("Transcribing dashboard audio...")
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self.stt.transcribe, audio_path)
            
            if text and len(text.strip()) >= 2:
                await self._process(text)
            else:
                logger.info("Dashboard audio transcription was empty.")
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

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
        user_id = self.ws.current_user_id.get() or 1
        
        # Reset barge-in state since we are now actively processing a new query
        self.barge_in_triggered = False
        
        logger.info("[User %s] Heard: %s", user_id, text)
        await self.ws.send_heard(text, target_user_id=user_id)

        # ── Load memory context ────────────────────────────────────────────
        context = self.memory.get_context(user_id)

        # ── RAG Context Injection (New) ───────────────────────────────────
        relevant_facts = self.memory.search_facts(user_id, text)

        # ── Route intent → get response ───────────────────────────────────
        if self.is_interviewing.get(user_id, False):
            if any(w in text.lower() for w in ["stop interview", "end interview", "stop the interview", "quit interview"]):
                self.is_interviewing[user_id] = False
                await self.ws.send_interview_mode(False, target_user_id=user_id)
                response = "Interview mode ended. Back to normal assistant mode."
                self.interviewer.reset()
            else:
                response = await self.interviewer.chat(text)
        else:
            response = await self.router.route(text, context, relevant_facts)
            if response == "__START_INTERVIEW__":
                self.is_interviewing[user_id] = True
                await self.ws.send_interview_mode(True, target_user_id=user_id)
                response = await self.interviewer.start_interview()

        # ── Shutdown command ───────────────────────────────────────────────────
        if response == "__SHUTDOWN__":
            await self._speak("Goodbye.")
            await self.ws.send_response("Shutting down.", target_user_id=user_id)
            import sys
            sys.exit(0)

        # ── Save to memory ────────────────────────────────────────────────
        self.memory.add_turn(user_id=user_id, user=text, assistant=response)

        # ── Background Memory Extractor ───────────────────────────────────
        from core.memory_extractor import extract_facts_background
        asyncio.create_task(extract_facts_background(text, self.memory))

        # ── Send to dashboard ─────────────────────────────────────────────
        await self.ws.send_response(response, target_user_id=user_id)

        # ── Send updated memory facts ─────────────────────────────────────
        facts = list(self.memory.get_all_facts(user_id).values())
        if facts:
            await self.ws.send_memory(facts, target_user_id=user_id)

        # ── Speak response ────────────────────────────────────────────────────
        logger.info("Responding: %s", response)
        
        duration = await self._speak(response)
        
        if duration > 0:
            await asyncio.sleep(duration)
            
        await asyncio.sleep(0.5)
        
        # Auto-listen if Charlie asked a question
        if '?' in response:
            logger.info("Response contains a question. Auto-listening for reply.")
            self.expecting_reply = True