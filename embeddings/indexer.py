import faiss
import pickle
import os
import numpy as np

class ImageIndexer:
    def __init__(self, dim=512, index_path="data/faiss.index", meta_path="data/gallery_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embedding, meta):
        self.index.add(np.array([embedding]))
        self.metadata.append(meta)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array([query_embedding]), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append((self.metadata[idx], float(dist)))
        return results
