from ultralytics import YOLO
import os

plate_model = YOLO("models/number_plate_detector_yolov8n.pt")

def detect_plate(vehicle_img):
    """
    Detect number plate within the cropped vehicle image.
    Returns cropped plate image and bbox within vehicle image.
    """
    results = plate_model(vehicle_img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_crop = vehicle_img[y1:y2, x1:x2]
            return plate_crop, (x1, y1, x2, y2)
    return None, None
