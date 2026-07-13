import base64
import io
import httpx
from PIL import ImageGrab

def test_moondream():
    screenshot = ImageGrab.grab()
    screenshot = screenshot.resize((screenshot.width // 2, screenshot.height // 2))
    buffer = io.BytesIO()
    screenshot.save(buffer, format="JPEG", quality=70)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Test 1: with num_predict
    r1 = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "moondream",
            "prompt": "Describe this screen",
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 150},
        },
        timeout=60
    )
    print("Test 1:", r1.json())

    # Test 2: without options
    r2 = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "moondream",
            "prompt": "Describe this screen",
            "images": [img_b64],
            "stream": False,
        },
        timeout=60
    )
    print("Test 2:", r2.json())

if __name__ == "__main__":
    test_moondream()
