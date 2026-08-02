from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, local model

def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Converts each chunk's text into a vector (list of numbers)
    that captures its meaning. Similar meaning = similar vector.
    Batch processing = faster than embedding one at a time.
    """
    texts = [c["text"] for c in chunks]
    embeddings = _model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings  # shape: (num_chunks, embedding_dim)