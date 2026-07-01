# Charlie v2

Personal AI voice assistant — offline-first, online-capable.

**Stack:** faster-whisper · OpenWakeWord · Piper TTS · Ollama · DuckDuckGo · pywhatkit · Playwright · YOLOv8

---

## Features

- Wake word activation (offline)
- Speech-to-text via faster-whisper (offline)
- Local LLM via Ollama — works without internet (offline)
- Web research via DuckDuckGo + LLM summarization (online)
- WhatsApp messaging via WhatsApp Web (online)
- Browser automation — open sites, navigate (online)
- Screen capture + OCR description (offline)
- Camera feed + YOLOv8 object detection (offline)
- SQLite memory — context across sessions

---

## Setup

### 1. Clone & create virtualenv

```bash
git clone https://github.com/YOUR_USERNAME/Charlie-v2
cd Charlie-v2
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Install external binaries

**Ollama** (local LLM server)
```
Download: https://ollama.ai
ollama pull qwen2.5:1.5b
```

**Piper TTS** (voice synthesis)
```
Download binary: https://github.com/rhasspy/piper/releases
Download voice model: https://huggingface.co/rhasspy/piper-voices
  → en_US-lessac-high.onnx + en_US-lessac-high.onnx.json
Place both files in: models/tts/
```

**Playwright browsers**
```bash
python -m playwright install chromium
```

**Tesseract OCR** (for screen/document reading)
```
Windows installer: https://github.com/UB-Mannheim/tesseract/wiki
Install to default path — config.py handles it automatically
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run

```bash
# Make sure Ollama is running first:
ollama serve

# Then start Charlie:
python main.py
```

---

## Contacts (WhatsApp)

Create `data/contacts.json` (gitignored — never committed):

```json
{
  "raj": "+91XXXXXXXXXX",
  "mom": "+91XXXXXXXXXX"
}
```

---

## Project Structure

```
Charlie-v2/
├── main.py              # Entry point
├── config.py            # All settings
├── requirements.txt
├── .env.example         # Config template
│
├── core/
│   ├── assistant.py     # Main loop orchestrator
│   ├── intent_router.py # Routes input to correct handler
│   └── memory.py        # SQLite short + long-term memory
│
├── speech/
│   ├── wake_word.py     # OpenWakeWord
│   ├── stt.py           # faster-whisper
│   └── tts.py           # Piper TTS
│
├── llm/
│   ├── local_llm.py     # Ollama wrapper
│   └── cloud_llm.py     # Claude API (optional)
│
├── research/
│   └── web_search.py    # DuckDuckGo search
│
├── actions/
│   ├── whatsapp.py      # pywhatkit
│   ├── browser.py       # Playwright
│   └── system.py        # Volume, apps, clipboard
│
├── vision/
│   ├── screen_capture.py
│   ├── camera.py        # YOLOv8
│   └── ocr.py
│
└── data/                # Gitignored
    ├── memory.db
    ├── contacts.json
    └── logs/
```

---

## Build Phases

| Phase | What | Status |
|-------|------|--------|
| 1 | STT + Local LLM + TTS pipeline | 🔨 |
| 2 | Memory + Intent routing | 🔨 |
| 3 | Web search + research | 🔨 |
| 4 | WhatsApp + Browser actions | 🔨 |
| 5 | Vision (camera + OCR) | 🔨 |

---

## Privacy

No personal data is committed to this repo. The following are gitignored:
- `.env` (API keys)
- `data/memory.db` (conversation history)
- `data/contacts.json` (phone numbers)
- `models/` (LLM weights)
- `User_Data/` (WhatsApp browser session)
