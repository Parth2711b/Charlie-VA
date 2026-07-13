"""
core/memory_extractor.py
Runs asynchronously in the background to extract long-term facts from the user's speech.
"""
import asyncio
import logging
from llm.local_llm import get_answer_llm

logger = logging.getLogger("Charlie.extractor")

async def extract_facts_background(user_text: str, memory):
    """
    Analyzes user_text to see if there is a permanent fact to remember.
    If so, saves it to ChromaDB + SQLite and updates the dashboard.
    """
    # Extremely strict prompt to avoid hallucinated memory
    prompt = (
        "You are a strict data extractor. Analyze the user's statement. "
        "If the user is stating a personal fact, preference, relationship, or detail "
        "that a personal AI assistant should remember permanently, extract it. "
        "Format your response exactly as: KEY: VALUE\n"
        "If there is no permanent personal fact to remember, you MUST output exactly: NONE\n\n"
        "Examples:\n"
        "User: My favorite color is blue\nOutput: favorite_color: blue\n\n"
        "User: What's the weather today?\nOutput: NONE\n\n"
        f"User: {user_text}\nOutput: "
    )
    
    try:
        llm = get_answer_llm()
        response = await llm._call([{"role": "user", "content": prompt}])
        response = response.strip()
        
        if response.upper() != "NONE" and ":" in response:
            parts = response.split(":", 1)
            key = parts[0].strip().lower().replace(" ", "_")
            val = parts[1].strip()
            
            # Sanity checks
            if 0 < len(key) < 40 and 0 < len(val) < 150:
                # Avoid saving random conversational junk as keys
                if "user" not in key and "output" not in key:
                    memory.save_fact(key, val)
                    logger.info("Background extractor learned a new fact -> %s: %s", key, val)
                    
                    # Send updated facts to dashboard so the UI card updates instantly
                    from core import websocket_bridge as ws
                    facts = list(memory.get_all_facts().values())
                    if facts:
                        asyncio.create_task(ws.send_memory(facts))
                        
    except Exception as e:
        logger.error("Memory extractor failed: %s", e)
