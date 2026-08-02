from loader import load_documents

docs = load_documents("test_docs")
print(len(docs))
for d in docs:
    print(d["filename"], "->", d["text"][:100])