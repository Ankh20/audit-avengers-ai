"""
retriever.py
Loads local policy documents, splits them into chunks, and retrieves
the most relevant chunks for a given query using keyword overlap scoring.
No vector database required.
"""

import os
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Supported plain-text extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}

CHUNK_SIZE = 400        # words per chunk
CHUNK_OVERLAP = 50      # word overlap between consecutive chunks

# Minimum Jaccard similarity (stopword-filtered) for a chunk to be considered
# relevant. Queries with no chunks above this are treated as out-of-scope.
# Calibrated: compliance queries score 0.01–0.08; off-topic score 0.0.
RETRIEVAL_THRESHOLD = 0.01

# Common English stopwords — excluded from Jaccard scoring so that
# function words ("a", "be", "when", "should") don't dilute the signal.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "i", "my", "we", "our",
    "you", "your", "he", "she", "they", "their", "not", "no", "nor",
    "so", "yet", "both", "either", "when", "where", "who", "which",
    "what", "how", "if", "then", "than", "also", "any", "all", "each",
    "more", "most", "other", "such", "into", "through", "about", "after",
    "before", "between", "under", "over", "per", "up", "out", "must",
}


@dataclass
class Chunk:
    source: str     # original filename
    section: int    # chunk index within the file (0-based)
    text: str       # chunk content


def _read_file(path: str) -> str:
    """Read a file as plain text, ignoring undecodable bytes."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
        return ""


def load_chunks(doc_dir: str) -> list[Chunk]:
    """
    Walk doc_dir, read every supported file, and split into overlapping chunks.
    Skips files that are empty or contain only whitespace.
    """
    chunks: list[Chunk] = []

    if not os.path.isdir(doc_dir):
        logger.warning("Policy document directory not found: %s", doc_dir)
        return chunks

    for fname in sorted(os.listdir(doc_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        full_path = os.path.join(doc_dir, fname)

        # Skip zero-byte or whitespace-only files silently
        if os.path.getsize(full_path) == 0:
            logger.debug("Skipping empty file: %s", fname)
            continue

        text = _read_file(full_path)
        if not text.strip():
            continue

        words = text.split()
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        section = 0

        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]
            chunks.append(Chunk(
                source=fname,
                section=section,
                text=" ".join(chunk_words),
            ))
            section += 1

    logger.info("Loaded %d chunks from %d files in %s",
                len(chunks),
                len({c.source for c in chunks}),
                doc_dir)
    return chunks


def _tokenize(text: str) -> set[str]:
    """
    Lowercase word tokens, stripping punctuation and English stopwords.
    Removing stopwords ensures Jaccard similarity is driven by meaningful
    domain terms rather than common function words.
    """
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOPWORDS


def retrieve(query: str, chunks: list[Chunk], top_k: int = 3) -> list[Chunk]:
    """Return top_k relevant chunks. Convenience wrapper around retrieve_with_scores."""
    pairs = retrieve_with_scores(query, chunks, top_k)
    return [c for c, _ in pairs]


def retrieve_with_scores(
    query: str, chunks: list[Chunk], top_k: int = 3
) -> list[tuple[Chunk, float]]:
    """
    Return (chunk, jaccard_score) pairs for the top_k most relevant chunks.

    Only chunks whose score meets RETRIEVAL_THRESHOLD are returned.
    An empty list means the query is out-of-scope for the loaded documents.
    """
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, Chunk]] = []

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        union = query_tokens | chunk_tokens
        jaccard = len(query_tokens & chunk_tokens) / len(union) if union else 0.0
        scored.append((jaccard, chunk))

    scored.sort(key=lambda x: -x[0])

    # Only return chunks that clear the relevance threshold
    top = [(c, s) for s, c in scored[:top_k] if s >= RETRIEVAL_THRESHOLD]
    return top
