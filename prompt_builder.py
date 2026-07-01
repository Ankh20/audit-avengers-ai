"""
prompt_builder.py
Constructs the Claude prompt with injected policy context and
instructions for inline citations and a self-reported confidence score.
"""

from retriever import Chunk


SYSTEM_CONTEXT = """You are an expert Treasury regulatory compliance assistant named Audit Avenger.
Your job is to answer questions strictly based on the provided policy documents.
You must never fabricate information or cite sources that are not present below."""


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """
    Build a complete prompt that instructs Claude to:
      - Answer using only the supplied policy documents
      - Cite each source inline as [SOURCE N]
      - End with a numbered Sources list
      - Append a self-assessed confidence score (0.0 – 1.0)

    Args:
        query:  The user's compliance question.
        chunks: Retrieved policy document chunks.

    Returns:
        A fully formed prompt string ready for Claude.
    """
    if not chunks:
        context_block = "No policy documents are currently available."
    else:
        sections = []
        for i, chunk in enumerate(chunks, start=1):
            header = f"[SOURCE {i}: {chunk.source} — section {chunk.section}]"
            sections.append(f"{header}\n{chunk.text}")
        context_block = "\n\n---\n\n".join(sections)

    prompt = f"""{SYSTEM_CONTEXT}

==============================
POLICY DOCUMENTS
==============================
{context_block}

==============================
QUESTION
==============================
{query}

==============================
INSTRUCTIONS
==============================
1. Answer the question using ONLY the policy documents above.
2. Cite every claim inline using [SOURCE N] notation.
3. If the documents do not contain enough information to answer fully, say:
   "The provided documents do not fully address this question."
4. After your answer, add a blank line then write exactly:
   Sources:
   - [SOURCE 1]: <filename>
   - [SOURCE 2]: <filename>
   (list only sources you actually cited)
5. After the Sources section, add a blank line then write exactly:
   Confidence: X.XX
   where X.XX is a number from 0.00 to 1.00 representing how confident
   you are that the documents fully support your answer.
   Use low scores (< 0.60) if the documents are incomplete or ambiguous."""

    return prompt
