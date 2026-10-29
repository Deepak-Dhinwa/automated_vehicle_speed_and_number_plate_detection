# Automated Vehicle Speed & Number Plate Detection System

One-page summary: open `poster.html` in a browser.
Start: `.\run.ps1`, then open `http://127.0.0.1:8000`. The bare server address used to return 404. `/` is the page, and `/video_feed` is the raw stream.

This project is an **end-to-end Computer Vision pipeline** where I built a system that can:

- Detect vehicles in real-time video streams (CCTV or IP webcam)
- Detect number plates on vehicles
- Read number plate text using a fine-tuned OCR model
- Track vehicles frame to frame to calculate their speed

---

## Project Structure

project/
├── app/
│ ├── main.py
│ ├── utils.py
│ ├── models/
│ │ ├── vehicle_detection.py
│ │ ├── plate_detection.py
│ │ ├── ocr.py
│ │ ├── yolov8n.pt
│ │ ├── number_plate_detector_yolov8n.pt
│ │ └── trocr_numberplate_finetuned/
│ └── static/
├── requirements.txt
├── README.md
├── sort/
├── speed_and_number_plate_detection.ipynb
└── data/
└── number plate detector data/
└── yolo dataset/
├── images/
│ ├── train/
│ └── val/
└── labels/
├── train/
└── val/
└── data.yaml

---

### 1. Vehicle Detection

- **Model used**: YOLOv8n (pre-trained COCO)
- **Training**: Used off-the-shelf pre-trained model for vehicle classes
- **Results**:
  - Detects cars, trucks, buses in real-time with good confidence on GPU RTX 3050(my local pc GPU)
  - Sample output:
    ```
    0: 384x640 21 cars, 1 bus, 2 trucks, 13.5ms
    ```
---

### 2. Number Plate Detection

- **Model used**: YOLOv8n custom-trained
- **Dataset**: Collected ~222 cropped number plate images with labels, converted to YOLO format
- **Training**:
  - Trained for 50 epochs
  - Achieved validation mAP50 of ~0.95
- **Results**:
  - Detects number plates accurately on cropped vehicles
  - Sample output:
    ```
    image.jpg: 384x640 1 number_plate, 31.6ms
    ```

---

### 3. OCR – Number Plate Text Extraction

- **Model used**: TrOCR base (Transformer OCR) fine-tuned
- **Dataset**: US number plate cropped images with only plate number as label
- **Training**:
  - Fine-tuned for 3 epochs
  - Final training loss ~0.54
- **Results**:
  - Exact match accuracy: ~10-20%
  - Fuzzy match accuracy (Levenshtein similarity >70%): ~50-60%
  - Example:
    ```
    GT: KTU979, Prediction: KTU 979
    GT: WRLDTC, Prediction: WRLDTC
    ```

---

### 4. Vehicle Tracking and Speed Calculation

- **Algorithm used**: SORT (Simple Online Realtime Tracking)
- **Implementation**:
  - Tracks vehicle IDs frame-to-frame
  - Calculates speed using pixel displacement per frame and real-world calibration factor(sample factor used)
- **Results**:
  - Speed output is fluctuating slightly, needs smoothing in future
  - Sample speed output:(on sample video)
    ```
    ID: 3, Speed: 41 km/h
    ID: 5, Speed: 37 km/h
    ```

---

## Deployment

Run it locally with the existing `plate_detection_env` (this is the environment that matches the saved YOLO and TrOCR weights). From the `app` folder:

```bash
..\plate_detection_env\Scripts\python.exe -m uvicorn main:app --port 8000
```

Open `http://127.0.0.1:8000`. That page shows the detection stream. `http://127.0.0.1:8000/video_feed` is the raw stream, which is why opening only the server root used to say 404. The default source is `data/videos/2103099-uhd_3840_2160_30fps.mp4`. Set `VIDEO_SOURCE` to a file, a stream URL, or `0` for the webcam.

A short headless check from the project root:

```bash
plate_detection_env\Scripts\python.exe testing.py
```

- **Dockerized** the entire app with:
  - FastAPI server serving real-time detection and tracking
  - Models included in image
- **Azure Container Registry (ACR) + Azure Container Instance**
  - Tagged and pushed docker image to ACR
  - Deployed container instance from ACR and tested live IP webcam stream

---

## Final Results and Observations

| Component             | Model                     | Result summary                              |
|------------------------|---------------------------|----------------------------------------------|
| Vehicle Detection      | YOLOv8n COCO             | Good detection speed and accuracy           |
| Number Plate Detection | YOLOv8n custom-trained   | mAP50 ~0.95, fast inference                 |
| OCR                    | TrOCR fine-tuned         | Fuzzy match ~50-60%, exact match low        |
| Tracking + Speed       | SORT                     | Working but speed output fluctuates          |

---

## Improvements Planned

- Increase OCR accuracy by training on a larger diverse dataset including Indian plates
- Implement exponential smoothing on speed outputs to reduce fluctuations
- Integrate traffic rules violation detection module in future

