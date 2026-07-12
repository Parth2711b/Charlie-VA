"""
handlers/debug_handler.py
Reads the OS clipboard for tracebacks and explains them to the user using the local LLM.
"""
import logging
import pyperclip
from llm.local_llm import get_answer_llm

logger = logging.getLogger("Charlie.handler.debug")

async def handle(text: str) -> str:
    try:
        clipboard_content = pyperclip.paste().strip()
    except Exception as e:
        logger.error("Failed to access clipboard: %s", e)
        return "I couldn't access your clipboard. Make sure you copied the error!"
        
    if not clipboard_content:
        return "Your clipboard is empty! Please copy the error message first."
        
    # Truncate if the clipboard is insanely long
    if len(clipboard_content) > 3000:
        clipboard_content = clipboard_content[-3000:]
        
    logger.info("Debugging clipboard content...")
    
    prompt = (
        "The user's code just crashed, and they copied the following text from their terminal or editor. "
        "Explain exactly why it crashed in very simple terms, and tell them how to fix it. "
        "Keep your response concise and conversational, as it will be spoken aloud.\n\n"
        f"Clipboard Content:\n{clipboard_content}"
    )
    
    llm = get_answer_llm()
    # We pass an empty context because this is a one-off debug task
    response = await llm.chat(prompt, [])
    return response
