from loader import load_documents
from chunker import chunk_text
from embedder import embed_chunks

docs = load_documents("test_docs")
chunks = chunk_text(docs, chunk_size=50, overlap=10)
embeddings = embed_chunks(chunks)

print(embeddings.shape)   # should be (9, 384)
print(embeddings[0][:5])  # first 5 numbers of first chunk's vectors