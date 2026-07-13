"""
config.py - All settings in one place.
Override via .env file or environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")        # routing/intent only
OLLAMA_ANSWER_MODEL = os.getenv("OLLAMA_ANSWER_MODEL", "qwen2.5:3b") # general answers

CLOUD_LLM_ENABLED  = os.getenv("CLOUD_LLM_ENABLED", "false").lower() == "true"
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLOUD_MODEL        = os.getenv("CLOUD_MODEL", "claude-sonnet-4-6")

# ── Speech ────────────────────────────────────────────────────────────────────
WHISPER_MODEL      = os.getenv("WHISPER_MODEL", "base")        # base = good CPU balance
WHISPER_LANGUAGE   = os.getenv("WHISPER_LANGUAGE", "en")
TTS_VOICE          = os.getenv("TTS_VOICE", "en_US-lessac-high")  # Piper voice
TTS_SPEED          = float(os.getenv("TTS_SPEED", "1.0"))

# ── Research ──────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SEARCH_TIMEOUT     = int(os.getenv("SEARCH_TIMEOUT", "10"))

# ── Actions ───────────────────────────────────────────────────────────────────
WHATSAPP_DEFAULT_WAIT = int(os.getenv("WHATSAPP_DEFAULT_WAIT", "15"))  # seconds
BROWSER_HEADLESS      = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"

# ── Vision ────────────────────────────────────────────────────────────────────
YOLO_MODEL         = os.getenv("YOLO_MODEL", "yolov8n.pt")    # nano = fastest on CPU
CAMERA_INDEX       = int(os.getenv("CAMERA_INDEX", "0"))

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_DB_PATH     = os.getenv("MEMORY_DB_PATH", "data/memory.db")
MAX_CONTEXT_TURNS  = int(os.getenv("MAX_CONTEXT_TURNS", "5"))

# ── Network ───────────────────────────────────────────────────────────────────
def is_online() -> bool:
    """Quick connectivity check. Socket is properly closed after use."""
    import socket
    try:
        # Create a socket with its OWN timeout (not global)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)  # per-socket timeout, doesn't affect other sockets
        sock.connect(("8.8.8.8", 53))
        sock.close()  # explicitly close — no leak!
        return True
    except OSError:
        return False

