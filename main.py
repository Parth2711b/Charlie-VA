"""
Charlie v2 - Entry Point
Starts dashboard server, opens browser, then runs the assistant.
"""

import asyncio
import logging
import subprocess
import threading
import time
import webbrowser
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/logs/charlie.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Charlie")

# ── Dashboard server ───────────────────────────────────────────────────────────
DASHBOARD_DIR  = os.path.join(os.path.dirname(__file__), "dashboard")
DASHBOARD_PORT = 8080
DASHBOARD_URL  = f"http://localhost:{DASHBOARD_PORT}"

dashboard_process = None


def start_dashboard_server():
    """Start HTTP server for dashboard in background."""
    global dashboard_process
    try:
        dashboard_process = subprocess.Popen(
            ["python", "-m", "http.server", str(DASHBOARD_PORT)],
            cwd=DASHBOARD_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Dashboard server started at %s", DASHBOARD_URL)
    except Exception as e:
        logger.error("Failed to start dashboard server: %s", e)


def open_browser():
    """Open dashboard in browser after server is ready."""
    time.sleep(2)
    webbrowser.open(DASHBOARD_URL)
    logger.info("Dashboard opened in browser.")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    from core.assistant import Assistant

    # Start dashboard server
    start_dashboard_server()

    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    logger.info("Charlie v2 starting up...")

    assistant = Assistant()
    
    from handlers.spotify import spotify_sync_loop
    asyncio.create_task(spotify_sync_loop())
    
    try:
        await assistant.run()
    finally:
        # Clean up dashboard server on exit
        if dashboard_process:
            dashboard_process.terminate()
            logger.info("Dashboard server stopped.")
        
        # Force immediate exit to bypass ThreadPoolExecutor waiting on blocking audio threads
        os._exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass