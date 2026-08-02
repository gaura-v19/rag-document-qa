import numpy as np
import pickle
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.embeddings = None   # numpy array of all chunk vectors
        self.chunks = None       # matching list of chunk dicts (filename, index, text)

    def build(self, chunks: list[dict], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings

    def save(self, path: str = "cache/store.pkl"):
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def load(self, path: str = "cache/store.pkl") -> bool:
        if not Path(path).exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        # cosine similarity between query and every chunk vector
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)
        scores = np.dot(self.embeddings, query_vector) / norms

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            results.append({**self.chunks[i], "score": float(scores[i])})
        return results