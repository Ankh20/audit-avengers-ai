"""
escalation.py
Computes a composite confidence score from two independent signals:

  1. Retrieval relevance  — how well the retrieved chunks matched the query
                            (Jaccard similarity scores from retriever.py)
  2. Model certainty      — parsed from the model's self-reported score in
                            its response text, adjusted by uncertainty phrases

The two signals are combined as a weighted average, then hard-capped at 0.95
so the system never claims perfect certainty about regulatory guidance.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────
ESCALATION_THRESHOLD = 0.60
CONFIDENCE_CAP       = 0.95   # never claim 100% certainty

# ── Weights for composite score ────────────────────────────────────────────
# Retrieval quality matters more than self-reported model certainty because
# Nova tends to be over-confident when documents only partially match.
RETRIEVAL_WEIGHT  = 0.55
MODEL_CERT_WEIGHT = 0.45

# ── Phrases that signal the model is uncertain ─────────────────────────────
UNCERTAINTY_PHRASES = [
    "does not fully address",
    "not specified",
    "cannot determine",
    "unclear",
    "insufficient",
    "not mentioned",
    "not covered",
    "no information",
    "i don't know",
    "unable to find",
    "not provided",
    "not addressed",
    "outside the scope",
]

# ── Phrases that signal the model is confident ────────────────────────────
CERTAINTY_PHRASES = [
    "clearly states",
    "explicitly states",
    "as stated in",
    "according to",
    "the document specifies",
    "the policy states",
    "as required by",
    "the regulation requires",
]


def _parse_model_score(response: str) -> float | None:
    """
    Extract the numeric confidence score the model embedded in its response.
    Returns None if no score line is found.
    """
    match = re.search(
        r"confidence[:\s]+([01](?:\.\d{1,2})?)",
        response,
        re.IGNORECASE,
    )
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass
    return None


def _heuristic_model_certainty(response: str) -> float:
    """
    Estimate model certainty from language patterns when no explicit score
    is present. Returns a value in [0.30, 0.90].
    """
    text = response.lower()
    uncertainty_hits = sum(p in text for p in UNCERTAINTY_PHRASES)
    certainty_hits   = sum(p in text for p in CERTAINTY_PHRASES)
    # Start at 0.70, subtract for uncertainty signals, add for certainty signals
    score = 0.70 - (uncertainty_hits * 0.12) + (certainty_hits * 0.05)
    return max(0.30, min(0.90, score))


def _retrieval_score(chunk_scores: list[float]) -> float:
    """
    Convert a list of per-chunk Jaccard scores into a single retrieval
    quality score in [0.0, 1.0].

    Strategy:
    - Best chunk score drives the result (weighted 60%)
    - Mean of all chunk scores fills in the rest (40%)
    - Apply a sigmoid-like stretch so middling scores aren't penalised too hard
    """
    if not chunk_scores:
        return 0.0

    best = max(chunk_scores)
    mean = sum(chunk_scores) / len(chunk_scores)
    raw  = 0.6 * best + 0.4 * mean

    # Stretch: Jaccard on natural-language text rarely exceeds 0.25 even for
    # highly relevant chunks. Scale up so 0.15+ Jaccard → good retrieval.
    stretched = min(1.0, raw * 4.0)
    return stretched


def compute_confidence(response: str, chunk_scores: list[float]) -> float:
    """
    Compute a composite confidence score capped at CONFIDENCE_CAP (0.95).

    Args:
        response:     Full text response from the model.
        chunk_scores: List of Jaccard similarity scores for retrieved chunks
                      (from retriever.retrieve_with_scores).

    Returns:
        Float in [0.0, CONFIDENCE_CAP].
    """
    # Signal 1 — retrieval quality
    ret_score = _retrieval_score(chunk_scores)

    # Signal 2 — model certainty
    parsed = _parse_model_score(response)
    if parsed is not None:
        # Deflate self-reported scores slightly — models over-report confidence
        model_cert = parsed * 0.85
    else:
        model_cert = _heuristic_model_certainty(response)

    # Weighted composite
    composite = RETRIEVAL_WEIGHT * ret_score + MODEL_CERT_WEIGHT * model_cert

    # Hard cap — regulatory answers should never claim perfect certainty
    final = min(composite, CONFIDENCE_CAP)

    logger.debug(
        "Confidence — retrieval: %.3f, model_cert: %.3f, composite: %.3f, final: %.3f",
        ret_score, model_cert, composite, final,
    )
    return round(final, 4)


# ── Kept for backward compatibility with any direct callers ───────────────
def extract_confidence(response: str) -> float:
    """
    Legacy single-signal extractor. Prefer compute_confidence() for new code.
    Returns model certainty only, capped at CONFIDENCE_CAP.
    """
    parsed = _parse_model_score(response)
    score  = (parsed * 0.85) if parsed is not None else _heuristic_model_certainty(response)
    return min(round(score, 4), CONFIDENCE_CAP)


def should_escalate(confidence: float) -> bool:
    """Return True when the answer should be reviewed by a human."""
    return confidence < ESCALATION_THRESHOLD


def escalation_message() -> str:
    return (
        "⚠️ **Escalation Required** — Confidence is below the acceptable threshold. "
        "This response must be reviewed by a qualified compliance officer before acting on it."
    )
