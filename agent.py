"""
agent.py
Orchestrates the full Audit Avengers AI pipeline:
  1. Retrieve relevant policy document chunks
  2. Build a cited prompt
  3. Call Amazon Nova Pro via Bedrock
  4. Extract confidence and decide on escalation
  5. Log the interaction
  6. Return a structured result dict
"""

import logging
from retriever import Chunk, load_chunks, retrieve_with_scores
from prompt_builder import build_prompt
from bedrock_client import invoke_model_with_fallback
from escalation import compute_confidence, should_escalate, escalation_message
from audit_logger import log_interaction

logger = logging.getLogger(__name__)

POLICY_DIR = "./policy_docs"
TOP_K_CHUNKS = 3

# Chunks are loaded once at import time so the Streamlit app stays fast
_chunks: list[Chunk] = []


def _ensure_chunks_loaded() -> None:
    """Load chunks from disk if not already loaded."""
    global _chunks
    if not _chunks:
        _chunks = load_chunks(POLICY_DIR)
        if not _chunks:
            logger.warning(
                "No policy documents found in %s. "
                "Add .txt or .md files to that directory.",
                POLICY_DIR,
            )


def reload_documents() -> int:
    """Force a reload of policy documents. Returns the new chunk count."""
    global _chunks
    _chunks = load_chunks(POLICY_DIR)
    return len(_chunks)


def run_query(query: str, session_id: str = "default") -> dict:
    """
    Run a compliance query through the full agent pipeline.

    Args:
        query:      The user's compliance question.
        session_id: Optional session identifier for audit grouping.

    Returns:
        dict with keys:
            response      (str)   — Amazon Nova Pro's full answer
            confidence    (float) — 0.0–1.0 self-assessed score
            escalated     (bool)  — True if confidence < threshold
            sources       (list)  — filenames of retrieved chunks
            escalation_note (str | None) — human-readable warning if escalated
            error         (str | None)  — error message if the call failed
    """
    _ensure_chunks_loaded()

    # --- Retrieval ---
    scored_chunks = retrieve_with_scores(query, _chunks, top_k=TOP_K_CHUNKS)
    relevant_chunks = [c for c, _ in scored_chunks]
    chunk_scores    = [s for _, s in scored_chunks]
    sources = list(dict.fromkeys(c.source for c in relevant_chunks))  # deduplicated

    # --- Generation ---
    try:
        prompt = build_prompt(query, relevant_chunks)
        response, is_fallback = invoke_model_with_fallback(prompt)
        if is_fallback:
            return {
                "response": response,
                "confidence": 0.0,
                "escalated": True,
                "sources": sources,
                "escalation_note": escalation_message(),
                "error": "Bedrock unavailable — fallback response returned.",
            }
    except RuntimeError as e:
        logger.error("Agent pipeline error: %s", e)
        return {
            "response": None,
            "confidence": 0.0,
            "escalated": True,
            "sources": sources,
            "escalation_note": escalation_message(),
            "error": str(e),
        }

    # --- Confidence & Escalation ---
    confidence = compute_confidence(response, chunk_scores)
    escalated = should_escalate(confidence)

    # --- Audit Logging ---
    log_interaction(
        query=query,
        response=response,
        sources=sources,
        confidence=confidence,
        escalated=escalated,
        session_id=session_id,
    )

    return {
        "response": response,
        "confidence": confidence,
        "escalated": escalated,
        "sources": sources,
        "escalation_note": escalation_message() if escalated else None,
        "error": None,
    }
