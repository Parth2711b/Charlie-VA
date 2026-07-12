"""
handlers/interviewer_agent.py
A strict FAANG mock interviewer mode that evaluates user responses using a verified Answer Key.
"""
import logging
import json
import random
import os
from llm.local_llm import get_answer_llm

logger = logging.getLogger("Charlie.interviewer")

class InterviewerAgent:
    def __init__(self, memory):
        self.memory = memory
        self.llm = get_answer_llm()
        self.context = []
        self.current_question = None
        self._load_questions()
        
    def _load_questions(self):
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "leetcode.json")
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                self.questions = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load leetcode DB: {e}")
            self.questions = []

    def reset(self):
        self.context = []
        self.current_question = None
        
    async def start_interview(self) -> str:
        self.reset()
        if not self.questions:
            return "I couldn't find my database of interview questions."
            
        self.current_question = random.choice(self.questions)
        
        system_prompt = (
            "You are a strict FAANG technical interviewer. "
            "You are evaluating the user on the following Data Structures problem:\n"
            f"Title: {self.current_question['title']}\n"
            f"Description: {self.current_question['description']}\n\n"
            "This is the ONLY acceptable optimal solution:\n"
            f"{self.current_question['solution']}\n\n"
            "Do NOT invent your own solutions. Evaluate the user based ONLY on the provided solution. "
            "Do not give them the direct answer immediately. Give hints if they struggle. "
            "Keep your responses conversational, short, and to the point. No code blocks."
        )
        self.context.append({"role": "system", "content": system_prompt})
        
        greeting = f"Welcome to your mock interview. Your question is: {self.current_question['description']} How would you approach this?"
        self.context.append({"role": "assistant", "content": greeting})
        return greeting
        
    async def chat(self, user_input: str) -> str:
        logger.info("Interviewer Agent processing: %s", user_input)
        
        # We pass ONLY the interview context, NOT the global Charlie context!
        response = await self.llm.chat(user_input, self.context)
        
        self.context.append({"role": "user", "content": user_input})
        self.context.append({"role": "assistant", "content": response})
        return response
