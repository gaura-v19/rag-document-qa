from retriever import retrieve

CONFIDENCE_THRESHOLD = 0.35  # tune based on testing

def retrieve_with_retry(query: str, store, top_k: int = 5) -> list[dict]:
    """
    If top result's similarity score is too low, the query is likely
    vague/mismatched — rephrase it once and search again.
    """
    results = retrieve(query, store, top_k)

    if not results or results[0]["score"] < CONFIDENCE_THRESHOLD:
        reformulated = _reformulate(query)
        retry_results = retrieve(reformulated, store, top_k)
        if retry_results and retry_results[0]["score"] > results[0]["score"] if results else True:
            return retry_results

    return results

def _reformulate(query: str) -> str:
    # simple rule-based reformulation (can upgrade to LLM-based later)
    return f"Explain in detail: {query}"