"""
vision/camera.py — Real-time camera feed with YOLOv8 object detection.
Uses YOLOv8n (nano) for speed on CPU.
"""

import logging
import cv2
from ultralytics import YOLO
from config import YOLO_MODEL, CAMERA_INDEX

logger = logging.getLogger("Charlie.camera")


class Camera:
    def __init__(self):
        logger.info("Loading YOLO model: %s", YOLO_MODEL)
        self.model = YOLO(YOLO_MODEL)
        self.cap   = None

    def describe_scene(self) -> str:
        """Capture one frame, run YOLO, return description string."""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return "Camera not accessible."

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Couldn't capture from camera."

        results = self.model(frame, verbose=False)
        detected = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]
                conf   = float(box.conf[0])
                if conf > 0.4:
                    detected.append(label)

        if not detected:
            return "Camera is working but I don't see anything clearly."

        unique = list(dict.fromkeys(detected))  # preserve order, deduplicate
        items  = ", ".join(unique[:5])           # cap at 5 for TTS brevity
        return f"I can see: {items}."

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
