# test_vehicle_detection.py
import cv2
from app.models import vehicle_detection

cap = cv2.VideoCapture(r"D:\projects\Automated Vehicle Speed & Number Plate Detection System\data\videos\2103099-uhd_3840_2160_30fps.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = vehicle_detection.detect(frame)
    print(detections)

    for (x1, y1, x2, y2, vid) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, str(vid), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
