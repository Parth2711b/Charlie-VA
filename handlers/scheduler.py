"""
handlers/scheduler.py
Handles scheduling events and reading the schedule.
"""

import logging
from datetime import datetime, timedelta
import dateparser
from llm.local_llm import get_answer_llm
from core.schedule_db import schedule_db
from core import websocket_bridge as ws

logger = logging.getLogger("Charlie.handler.scheduler")

async def handle(text: str) -> str:
    t = text.lower()
    user_id = ws.current_user_id.get() or 1
    
    # 1. READ SCHEDULE
    if any(k in t for k in ["what is my schedule", "what's my schedule", "what do i have", "read my schedule", "my events"]):
        events = schedule_db.get_todays_schedule(user_id)
        if not events:
            return "You have nothing scheduled for today. Your day is completely free!"
        
        resp = "Here is your schedule for today. "
        for ev in events:
            dt = datetime.fromisoformat(ev['scheduled_time'])
            time_str = dt.strftime("%I:%M %p").lstrip("0")
            status = " (completed)" if ev['status'] == 'completed' else ""
            resp += f"At {time_str}, {ev['event_text']}{status}. "
            
        return resp.strip()

    # 2. MARK DONE (Voice)
    if any(k in t for k in ["mark done", "mark as done", "i finished", "i completed"]):
        # A bit complex to find WHICH task to mark done via voice.
        # We can just say:
        return "Please mark tasks as done using the schedule tab on your dashboard."

    # 3. ADD TO SCHEDULE
    # Use LLM to extract the task description and dateparser for datetime
    prompt = f"""Extract the event description from the user's text.
Do not include the time or date in the description. Just the action.
Format exactly as: EVENT: <description>
User text: {text}
Output: """

    llm = get_answer_llm()
    response = await llm._call([{"role": "user", "content": prompt}])
    
    event_desc = text  # fallback
    for line in response.split('\n'):
        if line.startswith("EVENT:"):
            event_desc = line.replace("EVENT:", "").strip()
            break
            
    # Try to parse the datetime from the text
    # dateparser handles "tomorrow at 5pm", "in 2 hours", etc.
    parsed_date = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
    
    if not parsed_date:
        return f"I understood you want to schedule '{event_desc}', but I couldn't figure out the exact time. Please specify a time like 'tomorrow at 3 PM'."
        
    if parsed_date < datetime.now():
        parsed_date += timedelta(days=1)
        
    schedule_db.add_event(user_id, event_desc, parsed_date)
    
    # Send update to dashboard
    import asyncio
    asyncio.create_task(ws.emit({"type": "schedule_update"}))
    
    time_str = parsed_date.strftime("%A at %I:%M %p")
    return f"I've added '{event_desc}' to your schedule for {time_str}."
