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
        self.contexts = {}
        self.current_questions = {}
        self._load_questions()
        
    def _get_user(self):
        from core import websocket_bridge as ws
        return ws.current_user_id.get() or 1
        
    def _load_questions(self):
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "leetcode.json")
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                self.questions = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load leetcode DB: {e}")
            self.questions = []

    def reset(self):
        uid = self._get_user()
        self.contexts[uid] = []
        self.current_questions[uid] = None
        
    async def start_interview(self) -> str:
        self.reset()
        uid = self._get_user()
        if not self.questions:
            return "I couldn't find my database of interview questions."
            
        self.current_questions[uid] = random.choice(self.questions)
        q = self.current_questions[uid]
        
        system_prompt = (
            "You are a strict FAANG technical interviewer. "
            "You are evaluating the user on the following Data Structures problem:\n"
            f"Title: {q['title']}\n"
            f"Description: {q['description']}\n\n"
            "This is the ONLY acceptable optimal solution:\n"
            f"{q['solution']}\n\n"
            "Do NOT invent your own solutions. Evaluate the user based ONLY on the provided solution. "
            "Do not give them the direct answer immediately. Give hints if they struggle. "
            "Keep your responses conversational, short, and to the point. No code blocks."
        )
        self.contexts[uid].append({"role": "system", "content": system_prompt})
        
        greeting = f"Welcome to your mock interview. Your question is: {q['description']} How would you approach this?"
        self.contexts[uid].append({"role": "assistant", "content": greeting})
        return greeting
        
    async def chat(self, user_input: str) -> str:
        uid = self._get_user()
        logger.info("[User %s] Interviewer processing: %s", uid, user_input)
        
        ctx = self.contexts.get(uid, [])
        response = await self.llm.chat(user_input, ctx)
        
        self.contexts.setdefault(uid, []).append({"role": "user", "content": user_input})
        self.contexts[uid].append({"role": "assistant", "content": response})
        return response
