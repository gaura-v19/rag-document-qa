def chunk_text(documents: list[dict], chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Splits each document's text into overlapping word-based chunks.
    chunk_size: words per chunk
    overlap: words repeated between consecutive chunks, so context
    isn't lost at chunk boundaries.
    Returns list of dicts: {filename, chunk_index, text}
    """
    all_chunks = []

    for doc in documents:
        words = doc["text"].split()
        filename = doc["filename"]
        chunk_index = 0
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)

            all_chunks.append({
                "filename": filename,
                "chunk_index": chunk_index,
                "text": chunk_str
            })

            chunk_index += 1
            start += chunk_size - overlap  # move forward, but overlap a bit

    return all_chunks