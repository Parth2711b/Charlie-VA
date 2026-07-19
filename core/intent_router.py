"""
core/intent_router.py

Routes user input to the correct handler using a keyword-first pipeline.
No double LLM call: heuristic keyword matching decides if web search is needed.

Steps: shutdown check, keyword intent match, live-data heuristic, LLM fallback.
Uses two Ollama model sizes: tiny router model and a larger answer model.
"""

import logging
import re
# pyrefly: ignore [missing-import]
import chromadb
from config import is_online, CLOUD_LLM_ENABLED

logger = logging.getLogger("Charlie.router")

# ── Shutdown triggers ──────────────────────────────────────────────────────────
SHUTDOWN_WORDS = [
    "shut down", "shutdown", "goodbye charlie", "bye charlie",
    "stop charlie", "exit charlie", "turn off", "power off",
]

# ── Intent → keyword patterns ──────────────────────────────────────────────────
# Each entry: (intent_name, list_of_keyword_substrings)
# Checked in order - first match wins.
INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("youtube",    ["youtube", "search youtube", "find on youtube", "play on youtube", "play video on youtube"]),
    ("arxiv",      ["search arxiv", "find papers on", "latest research on", "research about", "papers about", "papers on"]),
    ("interview",  ["start a mock interview", "interview me", "practice for my interview", "mock interview", "coding interview", "practice coding", "interview mode"]),
    ("debug",      ["why did it crash", "read the traceback", "explain this error", "what went wrong", "debug this"]),
    ("research",   ["read my pdfs", "ingest pdfs", "read my papers", "scan documents", "read my research"]),
    ("weather",    ["weather", "temperature", "forecast", "how hot", "how cold", "what's the temp"]),
    ("timer",      ["set a timer", "set timer", "timer for", "start a timer", "cancel timer", "stop timer", "remind me in"]),
    ("notes",      ["save a note", "take a note", "make a note", "note that", "write down", "show my notes", "read notes", "clear notes", "my notes"]),
    ("clear_memory", ["clear memory", "forget everything", "delete memory", "reset memory"]),
    ("whatsapp",   ["whatsapp", "send message", "message to", "text to", "send a message"]),
    ("calculator", ["calculator", "calculate", "compute", "how much is", "times", "plus", "minus", "divided by", "square root"]),
    ("schedule",   ["schedule", "remind me on", "add to my calendar", "book a meeting", "new event", "what is my schedule", "what's my schedule", "what do i have", "my events", "mark done"]),

    ("camera_vision",  ["look at", "what do you see", "camera", "what am i holding", "describe this"]),
    ("screen_vision",  ["screenshot", "what's on my screen", "what on my screen"]),
    ("screen_translate", ["translate screen", "translate this", "translate text", "what does this mean in english"]),
    ("screen_read",    ["read screen", "read this out", "read text"]),
    ("system",     ["volume up", "volume down", "mute", "open app", "clipboard", "paste", "copy"]),

    ("dashboard",  ["show me the map", "global map", "focus news", "focus map",
                    "focus charlie", "show news", "reset panels", "show dashboard", "focus on", "open the map", "open map", "change theme", "hacker theme", "jarvis theme", "cyberpunk theme", "switch theme"]),
    ("music",      ["play music", "play a song", "pause", "resume", "skip", "next track", "pause music", "resume music", "play song", "play artist"]),
    ("browser",    ["open ", "go to ", "visit ", "browse ", "load "]),
    ("web_search", ["search the web for", "look up information about", "find online", "google this", "compare statistics", "search google for", "look up"]),
    ("conversation", ["who are you", "what is your name", "who created you", "how are you", "hello", "hi charlie", "hey charlie", "thanks", "thank you", "good morning", "goodbye"])
]

# ── Live-data keywords - needs web search ─────────────────────────────────────
LIVE_DATA_KEYWORDS = [
    "latest", "current", "today", "now", "live", "trending",
    "news", "score", "result", "price", "stock", "market",
    "weather in",  # handled by weather handler, but just in case
    "2024", "2025", "2026",
    "who won", "what happened",
    # Add general knowledge triggers to force web search instead of hallucinating facts
    "who is", "who was", "what is", "what are", "when was", "where is", "how many",
    "founded", "history", "capital", "population",
    "compare", "statistics", "stats", "search", "look up", "information on", "info on", "how tall", "how old"
]

# ── Pure-offline keywords - never needs search ────────────────────────────────
OFFLINE_KEYWORDS = [
    "joke", "poem", "write", "story", "rhyme", "explain",
    "define", "what is ", "who is ", "how does",
    "help me", "what can you", "thank you", "thanks",
    "remind", "note", "calculate", "convert",
    "your name", "who are you", "introduce yourself",
]


def _is_shutdown(text: str) -> bool:
    t = text.lower().strip()
    return any(w in t for w in SHUTDOWN_WORDS)


def _keyword_match(text: str) -> str | None:
    # Deprecated. We now use self._semantic_match
    return None


def _needs_live_data(text: str) -> bool:
    """
    Fast heuristic check - does this query need live internet data?
    No LLM involved - just string matching using word boundaries.
    """
    t = text.lower()
    
    # If it matches live-data keywords, definitely search
    for kw in LIVE_DATA_KEYWORDS:
        if re.search(rf'\b{re.escape(kw.strip())}\b', t):
            return True
            
    # If it matches offline keywords, never search
    for kw in OFFLINE_KEYWORDS:
        if re.search(rf'\b{re.escape(kw.strip())}\b', t):
            return False
            
    return False


class IntentRouter:
    def __init__(self, memory=None):
        self.memory = memory
        from config import USE_GROQ_LLM
        if USE_GROQ_LLM:
            from llm.groq_llm import get_answer_llm
        else:
            from llm.local_llm import get_answer_llm
            
        from research.web_search import WebSearch
        from actions.whatsapp import WhatsAppAction
        from actions.system import SystemAction
        from vision.screen_capture import ScreenCapture
        from vision.camera import Camera

        # Answer model
        self.answer_llm = get_answer_llm()

        # Cloud LLM (optional)
        from typing import Optional, Any
        self.cloud_llm: Optional[Any] = None
        if CLOUD_LLM_ENABLED:
            from llm.cloud_llm import CloudLLM
            self.cloud_llm = CloudLLM()

        self.web_search = WebSearch()
        self.whatsapp   = WhatsAppAction()
        self.system     = SystemAction()
        self.screen     = ScreenCapture()
        self.camera     = Camera()

        # ── Semantic Router Initialization ─────────────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path="data/chroma")
        
        total_phrases = sum(len(p) for _, p in INTENT_PATTERNS)
        try:
            self.intent_collection = self.chroma_client.get_collection(name="intents")
            # If the number of phrases changed, re-train
            if self.intent_collection.count() != total_phrases:
                self.chroma_client.delete_collection(name="intents")
                self.intent_collection = self.chroma_client.create_collection(name="intents")
                self._seed_intents()
        except:
            self.intent_collection = self.chroma_client.create_collection(name="intents")
            self._seed_intents()

    def _seed_intents(self):
        """Trains the semantic router by converting INTENT_PATTERNS into vectors."""
        docs, metas, ids = [], [], []
        i = 0
        for intent, phrases in INTENT_PATTERNS:
            for phrase in phrases:
                docs.append(phrase)
                metas.append({"intent": intent})
                ids.append(f"intent_seed_{i}")
                i += 1
        
        # This triggers the embedding model automatically!
        if docs:
            self.intent_collection.add(documents=docs, metadatas=metas, ids=ids)
            logger.info("Semantic Router seeded with %d training phrases.", len(docs))

    def _semantic_match(self, text: str) -> str | None:
        """Find the most mathematically similar intent."""
        try:
            results = self.intent_collection.query(
                query_texts=[text],
                n_results=1
            )
            
            if results and results.get("distances") and results["distances"][0]:
                best_dist = results["distances"][0][0]
                best_intent = results["metadatas"][0][0]["intent"]
                
                # A distance closer to 0 means highly similar.
                # 4. If the closest intent is good enough (smaller distance = better match)
                if best_dist < 0.95:  # Tightened from 1.1 to avoid false positives
                    if best_intent in ["knowledge", "conversation"]:
                        # Let the LLM handle conversation/knowledge questions directly
                        return None
                    logger.info("Semantic match found: %s (Distance: %.2f)", best_intent, best_dist)
                    return best_intent
                    
                logger.info("No confident match. Closest was %s (Distance: %.2f)", best_intent, best_dist)
                return None
        except Exception as e:
            logger.error("Semantic match failed: %s", e)
            
        return None

    async def route(self, text: str, context: list, relevant_facts: list[str] | None = None) -> str:
        # ── 1. Shutdown ────────────────────────────────────────────────────────
        if _is_shutdown(text):
            return "__SHUTDOWN__"

        # ── 1.5. Multi-intent Splitting ─────────────────────────────────────────
        import re
        # Split by common conjunctions if they have spaces around them
        parts = [p.strip() for p in re.split(r'\b(?:and|then|also)\b', text, flags=re.IGNORECASE) if len(p.strip()) > 2]
        
        # If it's a compound sentence, check if it contains intents
        if len(parts) > 1:
            part_intents = []
            has_intent = False
            for part in parts:
                intent = self._semantic_match(part)
                part_intents.append((part, intent))
                if intent:
                    has_intent = True
            
            if has_intent:
                responses = []
                llm_parts = []
                for part, intent in part_intents:
                    if intent:
                        try:
                            resp = await self._dispatch(intent, part, context)
                            if resp:
                                responses.append(resp)
                        except Exception as e:
                            logger.error("Handler error for '%s': %s", intent, e)
                    else:
                        llm_parts.append(part)
                
                # If there are leftovers for the LLM
                if llm_parts:
                    leftover_text = " and ".join(llm_parts)
                    online = is_online()
                    if online and _needs_live_data(leftover_text):
                        try:
                            results = await self.web_search.search(leftover_text)
                            llm = self.cloud_llm or self.answer_llm
                            resp = await llm.answer_with_context(leftover_text, results, context, relevant_facts)
                            responses.append(resp)
                        except Exception:
                            resp = await self.answer_llm.chat(leftover_text, context, relevant_facts)
                            responses.append(resp)
                    else:
                        resp = await self.answer_llm.chat(leftover_text, context, relevant_facts)
                        responses.append(resp)
                        
                return " ".join(responses)

        # ── 2. Semantic Intent Dispatch (Fast vector math, no LLM) ─────────────
        intent = self._semantic_match(text)

        if intent:
            try:
                return await self._dispatch(intent, text, context)
            except Exception as e:
                logger.error("Handler error for '%s': %s", intent, e, exc_info=True)
                return "I ran into an issue with that. Please try again."

        # ── 3. Direct LLM answer - pick search vs chat via fast heuristic ──────
        online = is_online()

        if online and _needs_live_data(text):
            logger.info("Live data query - searching web for: %s", text[:50])
            try:
                results = await self.web_search.search(text)
                llm = self.cloud_llm or self.answer_llm
                return await llm.answer_with_context(text, results, context, relevant_facts)
            except Exception as e:
                logger.error("Search failed, falling back to LLM: %s", e)

        # ── 4. Pure LLM answer (offline-safe) ─────────────────────────────────
        logger.info("Direct LLM answer: %s", text[:50])
        return await self.answer_llm.chat(text, context, relevant_facts)

    async def _dispatch(self, intent: str, text: str, context: list) -> str:
        """Route an identified intent to the correct handler."""
        if intent == "youtube":
            from handlers.youtube import handle
            return await handle(text)
            
        elif intent == "arxiv":
            from handlers.arxiv_handler import handle
            return await handle(text, self.memory)
            
        elif intent == "interview":
            return "__START_INTERVIEW__"
            
        elif intent == "debug":
            from handlers.debug_handler import handle
            return await handle(text)
        elif intent == "research":
            from handlers.pdf_handler import handle
            return handle(self.memory)  # We pass memory so the handler can save chunks

        elif intent == "web_search":
            logger.info("Semantic intent triggered web search for: %s", text[:50])
            try:
                results = await self.web_search.search(text)
                llm = self.cloud_llm or self.answer_llm
                return await llm.answer_with_context(text, results, context, "")
            except Exception as e:
                logger.error("Web search intent failed: %s", e)
                return "I'm having trouble searching the web right now."

        elif intent == "timer":
            from handlers.timer import handle
            return await handle(text)

        elif intent == "weather":
            from handlers.weather import handle
            return await handle(text)

        elif intent == "notes":
            from handlers.notes import handle
            return await handle(text)

        elif intent == "schedule":
            from handlers.scheduler import handle
            return await handle(text)

        elif intent == "clear_memory":
            if self.memory:
                self.memory.clear_all_facts()
                self.memory.clear_context()
                from core import websocket_bridge as ws
                await ws.emit({"type": "memory", "facts": []})
                return "I have wiped my memory banks."
            return "Memory is not active."

        elif intent == "whatsapp":
            return await self.whatsapp.handle(text)

        elif intent == "browser":
            from handlers.web import handle
            return await handle(text)

        elif intent == "camera_vision":
            return await self.camera.capture_and_describe_llava(prompt=text)
            
        elif intent == "screen_vision":
            return await self.screen.capture_and_describe(task="describe")
            
        elif intent == "screen_translate":
            return await self.screen.capture_and_describe(task="translate")
            
        elif intent == "screen_read":
            return await self.screen.capture_and_describe(task="read")

        elif intent == "system":
            return await self.system.handle(text)

        elif intent == "calculator":
            from handlers.calculator import handle
            return handle(text)

        elif intent == "dashboard":
            return await self._handle_dashboard_command(text)

        elif intent == "music":
            from handlers.spotify import play_song, pause_music, resume_music, skip_track
            t = text.lower()
            if "pause" in t:
                return await pause_music()
            elif "resume" in t:
                return await resume_music()
            elif "skip" in t or "next" in t:
                return await skip_track()
            elif "play" in t:
                # Extract the query after "play"
                query = text
                if "play " in t:
                    query = text[t.find("play ") + 5:]
                
                # Clean up query
                query = query.lower().replace("on spotify", "").strip()
                # Remove punctuation to help search accuracy
                import re
                query = re.sub(r'[^\w\s]', '', query)
                
                # If 'by' is present, use Spotify's strict advanced search
                if " by " in query:
                    parts = query.split(" by ", 1)
                    query = f"track:{parts[0].strip()} artist:{parts[1].strip()}"
                
                if query:
                    return await play_song(query)
                else:
                    return await resume_music()
                    
        return "I'm not sure how to handle that."

    async def _handle_dashboard_command(self, text: str) -> str:
        from core import websocket_bridge as ws
        t = text.lower()

        if "global" in t and "map" in t:
            await ws.emit({"type": "focus_panel", "panel": "map"})
            await ws.emit({"type": "map_region", "region": "global"})
            return "Switching to global map."

        elif "india" in t and "map" in t:
            await ws.emit({"type": "focus_panel", "panel": "map"})
            await ws.emit({"type": "map_region", "region": "india"})
            return "Switching to India map."

        elif "news" in t:
            await ws.emit({"type": "focus_panel", "panel": "news"})
            return "Focusing on news feed."

        elif "charlie" in t and "focus" in t:
            await ws.emit({"type": "focus_panel", "panel": "charlie"})
            return "Focusing on Charlie panel."

        elif "reset" in t or "normal" in t:
            await ws.emit({"type": "reset_panels"})
            return "Resetting panels to default."
            
        elif "stop" in t and ("video" in t or "youtube" in t):
            await ws.emit({"type": "stop_video"})
            return "Stopping video."
            
        elif "focus on" in t:
            cities = ["mumbai", "delhi", "bangalore", "pune", "chennai", "kolkata", "hyderabad", "jaipur", "ahmedabad", "surat"]
            for city in cities:
                if city in t:
                    await ws.emit({"type": "focus_panel", "panel": "map"})
                    await ws.emit({"type": "map_city", "city": city})
                    return f"Zooming map to {city.capitalize()}."
            # Fallback if city not found
            await ws.emit({"type": "focus_panel", "panel": "map"})
            return "Focusing on the map."
            
        elif "theme" in t:
            if "hacker" in t or "matrix" in t:
                await ws.emit({"type": "set_theme", "theme": "hacker"})
                return "Switching to Hacker theme."
            elif "jarvis" in t or "iron man" in t:
                await ws.emit({"type": "set_theme", "theme": "jarvis"})
                return "Switching to Jarvis theme."
            elif "cyberpunk" in t:
                await ws.emit({"type": "set_theme", "theme": "cyberpunk"})
                return "Switching to Cyberpunk theme."
            elif "ember" in t or "fire" in t:
                await ws.emit({"type": "set_theme", "theme": "ember"})
                return "Switching to Ember theme."
            elif "default" in t or "normal" in t or "clear" in t:
                await ws.emit({"type": "set_theme", "theme": "default"})
                return "Switching to default theme."
            else:
                return "I have Hacker, Jarvis, Cyberpunk, and Ember themes available. Which one would you like?"

        return "I'm not sure how to adjust the dashboard for that."