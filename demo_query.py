import numpy as np
import faiss
from embeddings.clip_encoder import embed_image, embed_text

index = faiss.read_index("data/live_index.faiss")
with open("data/live_paths.txt", "r") as f:
    gallery_paths = f.read().splitlines()

text_query = "A person with curly hair wearing a dark t-shirt with text"
query_vec = embed_text(text_query).astype('float32').reshape(1, -1)
faiss.normalize_L2(query_vec) 

distances, indices = index.search(query_vec, k=3)
percentages = (distances[0] * 100).round(2)

print("Text query:")
for idx, score in zip(indices[0], percentages):
    print({"path": gallery_paths[idx]}, f"{score:.2f}% match")

image_query_path = "data/test_data/person.jpg"
query_vec = embed_image(image_query_path).astype('float32').reshape(1, -1)
faiss.normalize_L2(query_vec)

distances, indices = index.search(query_vec, k=3)
percentages = (distances[0] * 100).round(2)

print("\nImage query:")
for idx, score in zip(indices[0], percentages):
    print({"path": gallery_paths[idx]}, f"{score:.2f}% match")
