"""Print vehicle counts for the first few frames of the bundled sample video.

Run from this folder with the existing environment:

    plate_detection_env\Scripts\python.exe testing.py
"""
import os

import cv2

from app.models.vehicle_detection import detect_and_track

VIDEO = os.path.join(
    os.path.dirname(__file__),
    "data",
    "videos",
    "2103099-uhd_3840_2160_30fps.mp4",
)

cap = cv2.VideoCapture(VIDEO)
frames = 0
total = 0
while frames < 3:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1
    height, width = frame.shape[:2]
    if width > 960:
        scale = 960 / width
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    detections = detect_and_track(frame)
    total += len(detections)
    print(f"frame {frames}: {len(detections)} vehicles")

cap.release()
print(f"done frames={frames} detections={total}")
