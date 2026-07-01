"""
bedrock_client.py
Calls Amazon Bedrock using the Converse API (works with Nova, Claude, and
any other Bedrock model that supports Converse).

Active model: Amazon Nova Pro — confirmed accessible in this account.
Fallback:     Returns a stub response so the app keeps running if Bedrock
              is unavailable (e.g., no network, expired creds).
"""

import boto3
import logging

logger = logging.getLogger(__name__)

# ── Model selection ────────────────────────────────────────────────────────
# Nova Pro: best reasoning quality among confirmed-accessible models.
# Swap to nova-lite for speed or nova-micro for lowest latency.
MODEL_ID = "amazon.nova-pro-v1:0"

# System prompt injected via the Converse API system field
SYSTEM_PROMPT = (
    "You are an expert Treasury regulatory compliance assistant named Audit Avenger. "
    "Answer questions strictly based on the policy documents provided in the user message. "
    "Always cite sources inline using [SOURCE N] notation. "
    "Never fabricate information or cite sources not present in the documents."
)

_client = None


def get_client():
    """Lazily create and cache the Bedrock Runtime client."""
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name="us-east-1")
    return _client


def invoke_model(prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """
    Send a prompt to the model via the Bedrock Converse API.

    Args:
        prompt:     The full prompt string (policy context + question).
        max_tokens: Maximum tokens in the response.
        temperature: 0.0 = deterministic.

    Returns:
        The assistant's text response.

    Raises:
        RuntimeError: Wraps any Bedrock exception with a clean message.
    """
    try:
        response = get_client().converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[
                {"role": "user", "content": [{"text": prompt}]}
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )
        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        logger.error("Bedrock Converse call failed: %s", e)
        raise RuntimeError(f"Bedrock call failed: {e}") from e


def invoke_model_with_fallback(prompt: str, max_tokens: int = 1024) -> tuple[str, bool]:
    """
    Attempt to invoke the model; return a fallback stub on failure.

    Returns:
        (response_text, is_fallback)
        is_fallback is True when the real model could not be reached.
    """
    try:
        return invoke_model(prompt, max_tokens=max_tokens), False
    except RuntimeError as e:
        logger.warning("Using fallback response. Reason: %s", e)
        fallback = (
            "⚠️ **Bedrock Unavailable** — The AI model could not be reached. "
            "Please check your AWS credentials and network connection, then try again.\n\n"
            f"Technical detail: {e}"
        )
        return fallback, True


# Keep backward-compatible alias used by agent.py
invoke_claude = invoke_model
