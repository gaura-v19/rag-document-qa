from loader import load_documents
from chunker import chunk_text
from embedder import embed_chunks, _model
from vector_store import VectorStore

docs = load_documents("test_docs")
chunks = chunk_text(docs, chunk_size=50, overlap=10)
embeddings = embed_chunks(chunks)

store = VectorStore()
store.build(chunks, embeddings)
store.save()

# test search with a query related to your test docs' content
query = "why is curiosity important"
query_vector = _model.encode([query])[0]
results = store.search(query_vector, top_k=3)

for r in results:
    print(r["filename"], r["chunk_index"], round(r["score"], 3), "->", r["text"][:80])