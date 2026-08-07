from loader import load_documents
from chunker import chunk_text
from embedder import embed_chunks
from vector_store import VectorStore
from retriever import retrieve

docs = load_documents("test_docs")
chunks = chunk_text(docs, chunk_size=50, overlap=10)
embeddings = embed_chunks(chunks)

store = VectorStore()
store.build(chunks, embeddings)

results = retrieve("how do cities become sustainable", store, top_k=3)

for r in results:
    print(r["filename"], r["chunk_index"], round(r["score"], 3), "->", r["text"][:80])