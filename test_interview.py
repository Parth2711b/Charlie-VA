import asyncio
import logging
from handlers.interviewer_agent import InterviewerAgent

# Disable verbose logging to just see the chat output
logging.basicConfig(level=logging.ERROR)

async def run_test():
    class DummyMemory:
        pass
        
    print("Initializing InterviewerAgent...")
    agent = InterviewerAgent(DummyMemory())
    
    print("\n--- STARTING INTERVIEW ---")
    greeting = await agent.start_interview()
    print("\nCHARLIE:\n" + greeting)
    
    print("\n--- CANDIDATE (BRUTE FORCE RESPONSE) ---")
    candidate_answer = "I would just use a double for loop to check every single possibility. I think that takes O(n^2) time."
    print("YOU:\n" + candidate_answer)
    
    print("\n--- CHARLIE'S EVALUATION ---")
    response = await agent.chat(candidate_answer)
    print("\nCHARLIE:\n" + response)

if __name__ == "__main__":
    asyncio.run(run_test())
