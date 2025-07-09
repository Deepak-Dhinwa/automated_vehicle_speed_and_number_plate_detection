from ultralytics import YOLO
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sort')))
from sort import Sort
import numpy as np
import cv2

# Loading YOLOv8 vehicle detection model
vehicle_model_path = "models/yolov8n.pt"

vehicle_model = YOLO(vehicle_model_path)

# Initializing SORT tracker
tracker = Sort()

def detect_and_track(frame):
    """
    Detect vehicles in the frame and track them using SORT.
    Returns list of (x1, y1, x2, y2, id).
    """
    results = vehicle_model(frame)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]:  # vehicle classes: car, motorcycle, bus, truck
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

    tracked = tracker.update(np.array(detections))

    results = []
    for d in tracked:
        x1, y1, x2, y2, obj_id = map(int, d)
        results.append((x1, y1, x2, y2, obj_id))

    return results
