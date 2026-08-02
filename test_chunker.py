from loader import load_documents
from chunker import chunk_text

docs = load_documents("test_docs")
chunks = chunk_text(docs, chunk_size=50, overlap=10)

print(len(chunks))
for c in chunks[:5]:
    print(c["filename"], c["chunk_index"], "->", c["text"][:80])