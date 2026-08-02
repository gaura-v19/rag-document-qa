from loader import load_documents
from chunker import chunk_text
from embedder import embed_chunks
from vector_store import VectorStore
from agent_controller import retrieve_with_retry
from generator import generate_answer

def main():
    store = VectorStore()

    if not store.load():
        print("No cache found. Building fresh index...")
        docs = load_documents("test_docs")
        chunks = chunk_text(docs, chunk_size=300, overlap=50)
        embeddings = embed_chunks(chunks)
        store.build(chunks, embeddings)
        store.save()
    else:
        print("Loaded cached index.")

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        results = retrieve_with_retry(query, store)
        generate_answer(query, results)

if __name__ == "__main__":
    main()