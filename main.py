"""
Charlie v2 — Main Entry Point
Wake word → STT → Intent → Action → TTS
"""

import asyncio
import logging
from core.assistant import Assistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/logs/Charlie.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Charlie")


async def main():
    logger.info("Charlie v2 starting up...")
    assistant = Assistant()
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
