"""
list_models.py
Tests every Claude (and optionally all) Bedrock foundation model by actually
invoking it with a tiny prompt. Reports which ones succeed vs. fail with
AccessDeniedException or other errors.

Run with:  python list_models.py
"""

import boto3
import json

bedrock     = boto3.client("bedrock",         region_name="us-east-1")
runtime     = boto3.client("bedrock-runtime", region_name="us-east-1")

TINY_BODY = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}],
})

def try_invoke(model_id: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    try:
        runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=TINY_BODY,
        )
        return True, "OK"
    except runtime.exceptions.AccessDeniedException:
        return False, "AccessDenied"
    except runtime.exceptions.ValidationException as e:
        return False, f"ValidationError: {e}"
    except Exception as e:
        err = type(e).__name__
        return False, err

# ── Fetch all models, keep only Claude inference-profile ones ──────────────
all_models = bedrock.list_foundation_models().get("modelSummaries", [])

# Build candidate list: Claude models + their us. prefixed variants
candidates = []
seen = set()

for m in all_models:
    mid = m["modelId"]
    name = m.get("modelName", "")
    inference = m.get("inferenceTypesSupported", [])

    if "claude" not in mid.lower():
        continue

    # Add the base ID
    if mid not in seen:
        candidates.append((mid, name, inference))
        seen.add(mid)

    # Add the cross-region us. prefix variant if INFERENCE_PROFILE
    if "INFERENCE_PROFILE" in inference:
        prefixed = f"us.{mid}"
        if prefixed not in seen:
            candidates.append((prefixed, name, ["INFERENCE_PROFILE (us.)"]))
            seen.add(prefixed)

print("=" * 70)
print("  TESTING CLAUDE MODELS — actual invocation")
print("=" * 70)

accessible = []
denied     = []
other      = []

for model_id, name, inference in candidates:
    ok, msg = try_invoke(model_id)
    status = "✅ ACCESSIBLE" if ok else ("🔴 AccessDenied" if "AccessDenied" in msg else f"⚠️  {msg}")
    print(f"  {status:<22}  {model_id}")
    if ok:
        accessible.append((model_id, name))
    elif "AccessDenied" in msg:
        denied.append(model_id)
    else:
        other.append((model_id, msg))

print()
print("=" * 70)
print(f"  RESULTS: {len(accessible)} accessible, {len(denied)} access-denied, {len(other)} other errors")
print("=" * 70)

if accessible:
    print("\n  ✅ Models you can invoke right now:")
    for mid, name in accessible:
        print(f"     {mid}  ({name})")
    print(f"\n  👉 Recommended: {accessible[0][0]}")
else:
    print("\n  ❌ No Claude models accessible. Check Bedrock model access in the AWS console.")
