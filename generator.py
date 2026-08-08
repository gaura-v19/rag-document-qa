from google import genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def generate_answer(query: str, chunks: list[dict]):
    """
    Builds a prompt from retrieved chunks, streams Gemini's answer,
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

    stream = client.models.generate_content_stream(
    model="gemini-3.6-flash",
    contents=prompt
)

    full_answer = ""
    for chunk in stream:
        token = chunk.text or ""
        full_answer += token
        print(token, end="", flush=True)

    print("\n\nSources:")
    for c in chunks:
        print(f"- {c['filename']} (chunk {c['chunk_index']}, score {c['score']:.2f})")

    return full_answer