import base64
import io
import httpx
from PIL import Image

def test_moondream(width, height):
    img = Image.new('RGB', (width, height), color = 'blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    r = httpx.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'moondream',
            'prompt': 'Describe this',
            'images': [img_b64],
            'stream': False
        },
        timeout=60
    )
    print(f"{width}x{height} -> {r.json().get('response', '')}")

for w, h in [(1920, 1080), (960, 540), (480, 270), (224, 224)]:
    try:
        test_moondream(w, h)
    except Exception as e:
        print(f"{w}x{h} -> Error: {e}")
