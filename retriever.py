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

# Supported plain-text extensions. Add more as needed.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}

CHUNK_SIZE = 400        # words per chunk
CHUNK_OVERLAP = 50      # word overlap between consecutive chunks


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

    Args:
        doc_dir: Path to the directory containing policy documents.

    Returns:
        List of Chunk objects ready for retrieval.
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

    logger.info("Loaded %d chunks from %s", len(chunks), doc_dir)
    return chunks


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, chunks: list[Chunk], top_k: int = 3) -> list[Chunk]:
    """
    Return the top_k most relevant chunks for a query using token overlap.

    Args:
        query:   The user's question.
        chunks:  Preloaded list of Chunk objects.
        top_k:   Number of chunks to return.

    Returns:
        List of up to top_k Chunk objects, best match first.
    """
    pairs = retrieve_with_scores(query, chunks, top_k)
    return [c for c, _ in pairs]


def retrieve_with_scores(
    query: str, chunks: list[Chunk], top_k: int = 3
) -> list[tuple[Chunk, float]]:
    """
    Same as retrieve() but also returns a normalised relevance score per chunk.

    Score is Jaccard similarity between query tokens and chunk tokens,
    capped to [0.0, 1.0].

    Returns:
        List of (Chunk, score) tuples, best match first.
    """
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    scored: list[tuple[float, Chunk]] = []

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        union = query_tokens | chunk_tokens
        # Jaccard similarity — avoids raw overlap inflating score on long chunks
        jaccard = len(query_tokens & chunk_tokens) / len(union) if union else 0.0
        scored.append((jaccard, chunk))

    scored.sort(key=lambda x: -x[0])
    top = [(c, s) for s, c in scored[:top_k] if s > 0.0]

    # Fall back to first top_k chunks if nothing overlaps
    if not top:
        top = [(c, 0.0) for _, c in scored[:top_k]]

    return top
