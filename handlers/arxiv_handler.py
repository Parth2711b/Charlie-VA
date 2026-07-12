"""
handlers/arxiv_handler.py
Extracts the topic from the voice command and triggers the ArXiv Agent.
"""
import logging
from research.arxiv_agent import ArxivAgent

logger = logging.getLogger("Charlie.handler.arxiv")

async def handle(text: str, memory) -> str:
    text = text.lower()
    topic = text
    
    # Try to extract just the topic part of the sentence
    prefixes = ["find papers on", "latest research on", "search arxiv for", "research about", "papers about", "papers on"]
    for prefix in prefixes:
        if prefix in text:
            topic = text.split(prefix)[-1].strip()
            break
            
    if not topic or topic == text:
        # Fallback if no prefix matched cleanly
        topic = text.replace("search arxiv", "").replace("find papers", "").strip()
        
    if not topic:
        return "What topic would you like me to research?"
        
    agent = ArxivAgent(memory)
    return await agent.research_topic(topic)
