import cv2
import numpy as np
import os
import glob
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
plate_cache = {}  # vid -> last plate text
plate_cache_frame = {}  # vid -> last frame index when OCR ran

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
    # Choose a source at runtime:
    # - set VIDEO_SOURCE to a filepath/URL (rtsp/http) OR a camera index ("0")
    # - default is a sample video (if found). To use webcam, set VIDEO_SOURCE=0 explicitly.
    video_source = os.environ.get("VIDEO_SOURCE")
    if not video_source:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            os.path.join(repo_root, "data", "videos", "2103099-uhd_3840_2160_30fps.mp4"),
        ]
        candidates.extend(glob.glob(os.path.join(repo_root, "data", "videos", "*.mp4")))
        candidates.extend(glob.glob(os.path.join(repo_root, "data", "*.mp4")))

        video_source = next((p for p in candidates if os.path.isfile(p)), None)
        if not video_source:
            raise RuntimeError(
                "No default sample video found. Set VIDEO_SOURCE to a video file/URL, "
                "or set VIDEO_SOURCE=0 to use the webcam."
            )
    try:
        source: object = int(video_source)
    except ValueError:
        source = video_source

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    # Performance knobs (set via env vars)
    # - MAX_WIDTH: resize input frames (default 960)
    # - FRAME_SKIP: run detection every Nth frame (default 2)
    # - OCR_EVERY: run OCR per vehicle id every N processed frames (default 10)
    max_width = int(os.environ.get("MAX_WIDTH", "960"))
    frame_skip = max(1, int(os.environ.get("FRAME_SKIP", "2")))
    ocr_every = max(1, int(os.environ.get("OCR_EVERY", "10")))
    jpeg_quality = int(os.environ.get("JPEG_QUALITY", "70"))

    frame_idx = 0
    last_vehicles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        if frame_idx % frame_skip == 0:
            last_vehicles = vehicle_detector(frame)
        vehicles = last_vehicles

        for (x1, y1, x2, y2, vid) in vehicles:
            vehicle_crop = frame[y1:y2, x1:x2]
            plate_crop, plate_bbox = plate_detector(vehicle_crop)
            plate_text = plate_cache.get(vid)

            if plate_crop is not None:
                last_ocr_frame = plate_cache_frame.get(vid, -10**9)
                if (frame_idx - last_ocr_frame) >= ocr_every:
                    plate_text = ocr_reader(plate_crop)
                    plate_cache[vid] = plate_text
                    plate_cache_frame[vid] = frame_idx
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

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(5, min(95, jpeg_quality))]
        _, jpeg = cv2.imencode('.jpg', frame, encode_params)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
