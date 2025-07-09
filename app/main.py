import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from models.vehicle_detection import detect_and_track
from models.plate_detection import detect_plate
from models.ocr import read_plate_text

app = FastAPI()

# Load models
vehicle_detector = detect_and_track
plate_detector = detect_plate
ocr_reader = read_plate_text

# Dictionary to store previous positions for speed calculation
previous_positions = {}

def calculate_speed(id, x1, y1, x2, y2, fps):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if id in previous_positions:
        prev_x, prev_y = previous_positions[id]
        pixel_distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
    else:
        pixel_distance = 0

    previous_positions[id] = (center_x, center_y)

    # Convert pixel/frame to pixel/sec
    pixel_per_second = pixel_distance * fps

    # Approximate scale factor (m/pixel). Adjust based on calibration
    scale_factor = 0.05
    speed_mps = pixel_per_second * scale_factor
    speed_kmph = speed_mps * 3.6

    return round(speed_kmph, 2)

def generate_frames():
    #cap = cv2.VideoCapture(r"D:\projects\Automated Vehicle Speed & Number Plate Detection System\data\videos\2103099-uhd_3840_2160_30fps.mp4")
    cap = cv2.VideoCapture("your webcam ip or sample video path")
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vehicles = vehicle_detector(frame)

        for (x1, y1, x2, y2, vid) in vehicles:
            vehicle_crop = frame[y1:y2, x1:x2]
            plate_crop, plate_bbox = plate_detector(vehicle_crop)
            plate_text = None

            if plate_crop is not None:
                plate_text = ocr_reader(plate_crop)
                px1, py1, px2, py2 = plate_bbox
                cv2.rectangle(vehicle_crop, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(vehicle_crop, plate_text, (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate speed
            speed = calculate_speed(vid, x1, y1, x2, y2, fps)

            # Draw bounding boxes and info on original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID:{vid} Speed:{speed}km/h Plate:{plate_text}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
