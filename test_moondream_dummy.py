import base64
import io
import httpx
from PIL import Image

def test_moondream():
    # Create a simple dummy image
    img = Image.new('RGB', (100, 100), color = 'red')
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    print("Testing moondream with options...")
    try:
        r1 = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": "What color is this image?",
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 150},
            },
            timeout=60
        )
        print("Response 1:", r1.json())
    except Exception as e:
        print("Error 1:", e)

    print("Testing moondream without options...")
    try:
        r2 = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": "What color is this image?",
                "images": [img_b64],
                "stream": False,
            },
            timeout=60
        )
        print("Response 2:", r2.json())
    except Exception as e:
        print("Error 2:", e)

if __name__ == "__main__":
    test_moondream()
