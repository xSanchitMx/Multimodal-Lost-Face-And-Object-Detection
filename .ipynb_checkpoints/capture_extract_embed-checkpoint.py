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

        # Embedding
        emb = embed_image(filename).astype("float32")
        faiss.normalize_L2(emb)

        if index.ntotal > 0:
            # Search for nearest neighbor
            D, I = index.search(emb, k=1)  # top-1 match
            best_score = D[0][0]
            best_match_idx = I[0][0]

            if best_match_idx != -1:
                matched_path = image_paths[best_match_idx]
                text = f"Match: {os.path.basename(matched_path)} | Score: {best_score:.2f}"
            else:
                text = "No match"
        else:
            text = "First entry"

        # Add embedding to index
        index.add(emb)
        image_paths.append(filename)

        # Draw rectangle + text on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


cap.release()
cv2.destroyAllWindows()

