"""
audit_logger.py
Appends every agent interaction to a newline-delimited JSON log file.
Each line is a self-contained JSON object — easy to parse, grep, or
import into a spreadsheet for auditors.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

LOG_FILE = "audit_log.jsonl"


def log_interaction(
    query: str,
    response: str,
    sources: list[str],
    confidence: float,
    escalated: bool,
    session_id: str = "default",
) -> None:
    """
    Append one interaction record to the audit log.

    Args:
        query:       The user's original question.
        response:    Amazon Nova Pro's full response text.
        sources:     List of document filenames cited.
        confidence:  Extracted confidence score (0.0 – 1.0).
        escalated:   Whether this response triggered escalation.
        session_id:  Optional identifier for grouping sessions.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "query": query,
        "response": response,
        "sources_cited": sources,
        "confidence": round(confidence, 4),
        "escalated": escalated,
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error("Failed to write audit log: %s", e)


def load_log() -> list[dict]:
    """
    Read all log entries from the audit log file.

    Returns:
        List of dicts, one per logged interaction (oldest first).
    """
    if not os.path.exists(LOG_FILE):
        return []

    entries = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed log line")
    except OSError as e:
        logger.error("Failed to read audit log: %s", e)

    return entries
