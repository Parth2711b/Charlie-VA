# test_pipeline.py
import asyncio
import subprocess
import wave
import tempfile
import pyaudio
import httpx
from faster_whisper import WhisperModel

# ── Constants ─────────────────────────────────────────────────────────────────
PIPER_BIN   = r"D:\COEP\charlie-v2\models\piper\piper.exe"
VOICE_MODEL = r"D:\COEP\charlie-v2\models\tts\en_US-lessac-high.onnx"
OLLAMA_URL  = "http://localhost:11434/api/chat"
LLM_MODEL   = "qwen2.5:1.5b"

CHUNK          = 1024
FORMAT         = pyaudio.paInt16
CHANNELS       = 1
RATE           = 16000
RECORD_SECONDS = 5

# ── STT ───────────────────────────────────────────────────────────────────────
def record_and_transcribe(whisper: WhisperModel) -> str:
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

    print("Listening... (5 seconds)")
    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    segments, _ = whisper.transcribe(tmp.name, language="en",
                                     beam_size=1, vad_filter=True)
    
    return " ".join(seg.text.strip() for seg in segments)

# ── LLM ───────────────────────────────────────────────────────────────────────
async def ask_llm(user_input: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Charlie, a personal AI voice assistant. "
                "Be concise - responses will be spoken aloud. "
                "No markdown, no bullet points, no asterisks."
            )
        },
        {"role": "user", "content": user_input}
    ]

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            OLLAMA_URL,
            json={
                "model":   LLM_MODEL,
                "messages": messages,
                "stream":  False,
                "options": {"temperature": 0.7, "num_predict": 150}
            }
        )
    return response.json()["message"]["content"].strip()

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    result = subprocess.run(
        [PIPER_BIN, "--model", VOICE_MODEL, "--output_file", "reply.wav"],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=30
    )
    if result.returncode == 0:
        import winsound
        winsound.PlaySound("reply.wav", winsound.SND_FILENAME)
    else:
        print("TTS error:", result.stderr.decode())

# ── Main Loop ─────────────────────────────────────────────────────────────────
async def main():
    print("Loading Whisper...")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")
    print("Charlie ready. Press Ctrl+C to stop.\n")

    while True:
        text = record_and_transcribe(whisper)
        print(f"You said: '{text}'")

        if not text.strip():
            print("Nothing heard, listening again...\n")
            continue

        if "stop" in text.lower() or "exit" in text.lower():
            speak("Goodbye.")
            break

        print("Thinking...")
        response = await ask_llm(text)
        print(f"Charlie: {response}\n")

        speak(response)

asyncio.run(main())