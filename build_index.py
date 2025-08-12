import os
import numpy as np
import faiss
from embeddings.clip_encoder import embed_image

gallery_dir = "data/gallery"
index_path = "data/index.faiss"
paths_file = "data/paths.txt"

image_paths = [os.path.join(gallery_dir, fname) for fname in os.listdir(gallery_dir)
               if fname.lower().endswith((".jpg", ".png", ".jpeg"))]

embeddings = []
print(f"Building index from {len(image_paths)} images.")

for path in image_paths:
    emb = embed_image(path).astype("float32")
    embeddings.append(emb)

embeddings = np.vstack(embeddings) 

faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim) 
index.add(embeddings)

faiss.write_index(index, index_path)
with open(paths_file, "w") as f:
    f.write("\n".join(image_paths))

print("Index built.")
