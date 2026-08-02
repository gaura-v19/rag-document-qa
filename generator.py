import ollama  # or replace with openai/other client

def generate_answer(query: str, chunks: list[dict]):
    """
    Builds a prompt from retrieved chunks, streams the LLM's answer,
    and appends citations at the end.
    """
    context = "\n\n".join(
        f"[{c['filename']} - chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""Answer the question using only the context below. Cite sources.

Context:
{context}

Question: {query}
"""

    stream = ollama.generate(model="llama3", prompt=prompt, stream=True)
    full_answer = ""
    for part in stream:
        token = part["response"]
        full_answer += token
        print(token, end="", flush=True)

    print("\n\nSources:")
    for c in chunks:
        print(f"- {c['filename']} (chunk {c['chunk_index']}, score {c['score']:.2f})")

    return full_answer