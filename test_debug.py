import asyncio
import logging
import pyperclip
from handlers import debug_handler

logging.basicConfig(level=logging.ERROR)

async def run_test():
    fake_error = """Traceback (most recent call last):
  File "main.py", line 4, in <module>
    print(10 / 0)
ZeroDivisionError: division by zero"""
    
    print("Copying fake ZeroDivisionError to clipboard...")
    pyperclip.copy(fake_error)
    
    print("\n--- SIMULATING USER ASKING 'WHY DID IT CRASH?' ---")
    response = await debug_handler.handle("why did it crash")
    
    print("\nCHARLIE'S EXPLANATION:\n" + response)

if __name__ == "__main__":
    asyncio.run(run_test())
