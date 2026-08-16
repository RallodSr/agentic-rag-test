"""Custom retrieval tool for the Data Retriever agent.

Implements a small RAG retrieval step over a local text file:
1. Load ``knowledge_base.txt`` and split it into paragraph chunks.
2. Embed every chunk once with the Gemini embedding model (cached in memory).
3. For each query, embed the query and rank chunks by cosine similarity.

If the embedding API is unavailable (quota / network), the tool falls back
to a simple keyword-overlap score so the pipeline still works end to end.
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base.txt")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "3"))

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "how", "when", "where", "why", "on", "in", "at", "of", "for", "to",
    "and", "or", "do", "does", "can", "may", "policy", "about",
}


def _load_chunks() -> list[str]:
    """Read the knowledge base and split it into paragraph chunks."""
    with open(KNOWLEDGE_BASE_PATH, encoding="utf-8") as f:
        text = f.read()
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@lru_cache(maxsize=1)
def _embedded_chunks() -> tuple[list[str], list[list[float]]]:
    """Embed all chunks once per process and cache the result."""
    chunks = _load_chunks()
    embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return chunks, embedder.embed_documents(chunks)


def _semantic_search(query: str, k: int) -> list[str]:
    chunks, vectors = _embedded_chunks()
    query_vec = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL).embed_query(query)
    ranked = sorted(
        zip(chunks, vectors), key=lambda cv: _cosine(query_vec, cv[1]), reverse=True
    )
    return [chunk for chunk, _ in ranked[:k]]


def _keyword_search(query: str, k: int) -> list[str]:
    """Fallback: rank chunks by overlap of non-stopword query terms."""
    terms = {
        w for w in re.findall(r"[a-zA-Z]+", query.lower()) if w not in _STOPWORDS
    }
    scored = [
        (sum(chunk.lower().count(t) for t in terms), chunk)
        for chunk in _load_chunks()
    ]
    scored.sort(key=lambda sc: sc[0], reverse=True)
    return [chunk for score, chunk in scored[:k] if score > 0]


@tool
def search_knowledge_base(query: str) -> str:
    """Search knowledge_base.txt and return the raw text chunks most
    relevant to the query. Always call this tool to look up information;
    never answer from your own knowledge. Returns snippets separated by
    '---', or 'NO_RELEVANT_INFORMATION_FOUND' when nothing matches."""
    try:
        results = _semantic_search(query, TOP_K)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on API errors
        print(f"[tools] semantic search unavailable ({exc}); using keyword search")
        results = _keyword_search(query, TOP_K)

    if not results:
        return "NO_RELEVANT_INFORMATION_FOUND"
    return "\n---\n".join(results)
