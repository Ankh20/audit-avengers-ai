"""
prompt_builder.py
Constructs the Amazon Nova Pro prompt with injected policy context and
instructions for inline citations and a self-reported confidence score.
"""

from retriever import Chunk


SYSTEM_CONTEXT = """You are an expert Treasury regulatory compliance assistant named Audit Avenger.
Your job is to answer questions strictly based on the provided policy documents.
You must never fabricate information or cite sources that are not present below."""


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """
    Build a complete prompt that instructs Amazon Nova Pro to:
      - Answer using only the supplied policy documents
      - Cite each source inline as [SOURCE N]
      - End with a numbered Sources list
      - Append a self-assessed confidence score (0.0 – 1.0)

    Args:
        query:  The user's compliance question.
        chunks: Retrieved policy document chunks.

    Returns:
        A fully formed prompt string ready for Amazon Nova Pro.
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
   Then list ONLY the sources you actually cited inline, one per line, like:
   - [SOURCE 1]: bsa_ctr_requirements.txt
   Do NOT include a source line if you did not cite that source number inline.
   Do NOT include blank bullet points or placeholder lines.
5. After the Sources section, add a blank line then write exactly:
   Confidence: X.XX
   where X.XX is a number from 0.00 to 1.00 representing how confident
   you are that the documents fully support your answer.
   Use a score below 0.60 ONLY if the documents are genuinely incomplete
   or the question cannot be answered from the provided content.
   Use 0.80 or higher if the documents clearly and directly answer the question."""

    return prompt
