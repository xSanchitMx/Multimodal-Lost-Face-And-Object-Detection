import cv2
from ultralytics import YOLO
import os
import time
import numpy as np
import faiss
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'embeddings')))
from clip_encoder import embed_image


model = YOLO("yolov8n.pt")

output_dir = "data/cropped_objects"
os.makedirs(output_dir, exist_ok=True)


dim = 512  # CLIP ViT-B/32 outputs 512-dim embeddings
index = faiss.IndexFlatIP(dim)
image_paths = []

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

        #Embedding and adding to Faiss
        emb = embed_image(filename).astype("float32")
        faiss.normalize_L2(emb)
        index.add(emb)
        image_paths.append(filename)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Live-Feed", frame)
    frame_count += 1

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


faiss.write_index(index, "data/live_index.faiss")
with open("data/live_paths.txt", "w") as f:
    f.write("\n".join(image_paths))

