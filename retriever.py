from embedder import _model
from vector_store import VectorStore

def retrieve(query: str, store: VectorStore, top_k: int = 5) -> list[dict]:
    """
    Embeds the query the same way chunks were embedded,
    then finds the top_k most similar chunks.
    """
    query_vector = _model.encode([query])[0]
    return store.search(query_vector, top_k=top_k)