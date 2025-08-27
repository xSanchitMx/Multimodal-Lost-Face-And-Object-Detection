import cv2
from ultralytics import YOLO
import os
import time

model = YOLO("yolov8n.pt")

output_dir = "data/cropped_objects"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = frame[y1:y2, x1:x2]
        timestamp = int(time.time() * 1000)
        filename = f"{output_dir}/crop_{frame_count}_{i}_{timestamp}.jpg"
        cv2.imwrite(filename, cropped)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
