"""
vision/camera.py - Real-time camera feed with YOLOv8 object detection.
Uses YOLOv8n (nano) for speed on CPU.
"""

import logging
import cv2
from ultralytics import YOLO
from config import YOLO_MODEL, CAMERA_INDEX

logger = logging.getLogger("Charlie.camera")


class Camera:
    def __init__(self):
        # Don't load YOLO here — it takes ~10 seconds and blocks startup.
        # We load it lazily on first use via _get_model().
        self._model = None
        self.cap    = None

    def _get_model(self):
        """Lazy-load YOLO model on first use."""
        if self._model is None:
            logger.info("Loading YOLO model: %s (first use)", YOLO_MODEL)
            self._model = YOLO(YOLO_MODEL)
        return self._model

    def describe_scene(self) -> str:
        """Capture one frame, run YOLO, return description string."""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return "Camera not accessible."

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Couldn't capture from camera."

        model = self._get_model()
        results = model(frame, verbose=False)
        detected = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                conf   = float(box.conf[0])
                if conf > 0.4:
                    detected.append(label)

        if not detected:
            return "Camera is working but I don't see anything clearly."

        unique = list(dict.fromkeys(detected))  # preserve order, deduplicate
        items  = ", ".join(unique[:5])           # cap at 5 for TTS brevity
        return f"I can see: {items}."

    async def capture_and_describe_llava(self, prompt: str = None) -> str:
        """Capture one frame, send to LLaVA for semantic understanding. Async to avoid blocking."""
        import base64
        import httpx
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return "Camera not accessible."

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Couldn't capture from camera."
            
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        
        final_prompt = prompt if prompt else "Briefly describe what you see in this camera feed in 1 or 2 short sentences. Focus on the main subject."
        
        try:
            # ASYNC client — doesn't freeze the event loop!
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llava:7b",
                        "prompt": final_prompt,
                        "images": [img_b64],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 100},
                    },
                )
            
            result = response.json().get("response", "").strip()
            logger.info("Camera LLaVA response: %s", result)
            return result if result else "I couldn't understand what the camera saw."
        except Exception as e:
            logger.error("Camera LLaVA error: %s", e)
            return "Failed to analyze camera feed with LLaVA."

    def start_live_feed(self):
        """Open a live window with YOLO detection overlaid. Press Q to quit."""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        logger.info("Live camera feed started. Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            annotated = results[0].plot()

            cv2.imshow("Charlie Vision", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
