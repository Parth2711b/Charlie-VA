"""
research/arxiv_agent.py
Queries the ArXiv API and synthesizes a summary of the latest research.
"""
import urllib.request
import xml.etree.ElementTree as ET
import logging
from llm.local_llm import get_answer_llm

logger = logging.getLogger("Charlie.research.arxiv")

class ArxivAgent:
    def __init__(self, memory):
        self.memory = memory
        self.llm = get_answer_llm()
        
    async def research_topic(self, topic: str) -> str:
        logger.info(f"Researching ArXiv for: {topic}")
        
        # 1. Format the URL for the ArXiv API (Top 3 results)
        query = topic.replace(" ", "+")
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"
        
        try:
            # 2. Fetch and parse the XML response
            response = urllib.request.urlopen(url).read()
            root = ET.fromstring(response)
            
            papers = []
            # ArXiv XML uses this namespace: {http://www.w3.org/2005/Atom}
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                papers.append({"title": title, "summary": summary})
                
                # Secretly save the abstracts to our RAG memory for later!
                self.memory.collection.upsert(
                    documents=[f"Abstract for {title}: {summary}"],
                    metadatas=[{"source": "ArXiv", "topic": topic}],
                    ids=[title]
                )
                
            if not papers:
                return f"I couldn't find any recent papers on {topic}."
                
            # 3. Build a prompt for the LLM to synthesize the data
            synthesis_prompt = f"You are a research assistant. Synthesize these 3 recent papers on '{topic}' into a 3-sentence summary of the current state of the art.\n\n"
            for p in papers:
                synthesis_prompt += f"Title: {p['title']}\nAbstract: {p['summary']}\n\n"
                
            # 4. Ask the LLM to generate the final response
            logger.info("Synthesizing research with LLM...")
            synthesis = await self.llm.chat(synthesis_prompt, [])
            
            return f"I found 3 papers on {topic}. Here is the summary: {synthesis}"
            
        except Exception as e:
            logger.error("ArXiv research failed: %s", e)
            return "I had trouble connecting to the research database."
