"""
core/memory_extractor.py
Runs asynchronously in the background to extract long-term facts from the user's speech.
"""
import asyncio
import logging
from config import USE_GROQ_LLM
if USE_GROQ_LLM:
    from llm.groq_llm import get_answer_llm
else:
    from llm.local_llm import get_answer_llm

logger = logging.getLogger("Charlie.extractor")

async def extract_facts_background(user_text: str, memory):
    """
    Analyzes user_text to see if there is a permanent fact to remember.
    If so, saves it to ChromaDB + SQLite and updates the dashboard.
    """
    prompt = (
        "Extract one permanent personal fact about the user from the following text. "
        "Return ONLY the fact in this format: KEY: VALUE\n"
        "If there is no permanent fact to remember, you MUST return ONLY the word: NONE\n\n"
        f"Text: {user_text}\n"
        "Output: "
    )
    
    try:
        llm = get_answer_llm()
        response = await llm._call([{"role": "user", "content": prompt}])
        response = response.strip()
        
        # Only take the first line to avoid multi-line hallucinations
        first_line = response.split('\n')[0].strip()
        
        if first_line.upper() != "NONE" and ":" in first_line:
            import re
            # Strictly match "key: value" format where key is just words/underscores
            match = re.match(r'^([a-zA-Z0-9_]+):\s*(.+)$', first_line)
            if match:
                key = match.group(1).lower()
                val = match.group(2).strip()
                
                # Sanity checks
                if 0 < len(key) < 40 and 0 < len(val) < 150:
                    if "user" not in key and "output" not in key and "key" not in key:
                        from core import websocket_bridge as ws
                        user_id = ws.current_user_id.get() or 1
                        
                        memory.save_fact(user_id, key, val)
                        logger.info("[User %s] Background extractor learned a new fact -> %s: %s", user_id, key, val)
                        
                        facts = list(memory.get_all_facts(user_id).values())
                        if facts:
                            asyncio.create_task(ws.send_memory(facts))
                    else:
                        logger.debug("Discarded hallucinated key: %s", key)
            else:
                logger.debug("Discarded improperly formatted fact: %s", first_line)
                        
    except Exception as e:
        logger.error("Memory extractor failed: %s", e)
