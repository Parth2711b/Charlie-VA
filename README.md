# Charlie v2

Personal AI voice assistant - offline-first, online-capable.

**Stack:** Groq STT/LLM · Edge TTS · ChromaDB (Semantic Routing) · Ollama · DuckDuckGo · Spotify API · Open-Meteo

---

## New Features

- **Dynamic Dashboard**: A beautiful glassmorphic web dashboard (http://localhost:8080)
- **Multi-tenant WebSocket**: Supports multiple users with independent chat histories and memory.
- **Semantic Intent Routing**: Uses ChromaDB and embedding models to mathematically route user commands to the correct handler (e.g. `play something` routes to the music module).
- **Public Tunnels**: Supports easy exposition via Cloudflare Tunnels for remote phone access.
- **Echo Cancellation**: Smart local microphone handling to prevent Charlie from triggering himself.

## Features

- Wake word activation via Vosk
- Lightning-fast Groq APIs for Speech-to-Text and LLM
- Local LLM fallback via Ollama
- Web research via DuckDuckGo + LLM summarization
- Spotify integration for music playback directly in the dashboard
- Interactive global maps and live weather data
- SQLite + ChromaDB memory - persistent context across sessions

---

## Setup

### 1. Clone & create virtualenv

```bash
git clone https://github.com/Parth2711b/Charlie-VA
cd Charlie-v2
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (GROQ_API_KEY, SPOTIFY_CLIENT_ID, etc.)
```

### 3. Run

```bash
python main.py
```
The server will start both a dashboard on port 8080 and a WebSocket bridge on port 8765.

---

## Project Structure

```
Charlie-v2/
├── main.py              # Entry point
├── config.py            # All settings
├── core/
│   ├── assistant.py     # Main loop orchestrator
│   ├── intent_router.py # ChromaDB semantic intent matching
│   ├── websocket_bridge.py # Multi-tenant WebSockets
│   └── memory.py        # SQLite short + long-term memory
│
├── speech/
│   ├── wake_word.py     # Vosk wake word
│   ├── groq_stt.py      # Groq Speech-to-text
│   └── tts.py           # Edge TTS
│
├── handlers/            # Intent Handlers
│   ├── spotify.py       # Spotify Search
│   ├── youtube.py       # YouTube Search
│   ├── weather.py       # Open-Meteo
│   └── web.py           # DuckDuckGo
│
├── dashboard/           # Frontend Web UI
│   ├── index.html       
│   └── index.css        # Glassmorphic themes
│
└── data/                # Gitignored
    ├── memory.db
    ├── chroma/          # Vector embeddings
    └── logs/
```

---

## Privacy

No personal data is committed to this repo. The following are gitignored:
- `.env` (API keys)
- `data/memory.db` (conversation history)
- `data/chroma/` (vector database)
- `.cache/` (Spotify auth)
