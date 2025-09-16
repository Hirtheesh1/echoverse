# backend/rewrite.py
import os
import time
from dotenv import load_dotenv
import requests
from .utils import chunk_text_by_chars, safe_trim

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_MODEL = os.getenv("HF_API_MODEL", "tiiuae/falcon-7b-instruct")
HF_API = os.getenv("HF_API", f"https://api-inference.huggingface.co/models/{HF_API_MODEL}")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Try modern and fallbacks for huggingface client
try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_INFERENCE_CLIENT_AVAILABLE = True
except Exception:
    InferenceClient = None
    HUGGINGFACE_INFERENCE_CLIENT_AVAILABLE = False

DEFAULT_CHUNK_MAX = 3000


def build_prompt_chunk(text: str, tone: str):
    return (
        f"Rewrite the following text while preserving meaning and factual content. "
        f"Apply a {tone.lower()} narrative tone. Improve flow and clarity while staying concise.\n\n"
        f"Original:\n{text}\n\nRewritten:"
    )


def _http_post_hf(prompt: str, max_new_tokens=300, temp=0.5, timeout=120):
    """Raw HTTP POST fallback. Returns (ok, status, body_text, headers)."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temp},
        "options": {"wait_for_model": True},
    }
    try:
        resp = requests.post(HF_API, headers=HEADERS, json=payload, timeout=timeout)
        status = resp.status_code
        try:
            body_text = resp.text
        except Exception:
            body_text = "<could not read response.text>"
        return resp.ok, status, body_text, dict(resp.headers)
    except Exception as e:
        return False, None, f"exception:{e}", {}


def call_hf_inference_once(prompt: str, max_new_tokens=300, temp=0.5, timeout=120):
    """
    Try InferenceClient (if available) first (compatible with different signatures).
    If any client error occurs (including StopIteration), capture it and fall back to raw HTTP POST.
    Returns success string OR failure tuple:
      (None, status, body_preview, headers, client_exc_info)
    """
    client_exc_info = None

    if HUGGINGFACE_INFERENCE_CLIENT_AVAILABLE:
        try:
            client = InferenceClient(token=HF_TOKEN)
            # Try modern signature first (inputs), fall back to prompt for older versions.
            try:
                result = client.text_generation(
                    model=HF_API_MODEL,
                    inputs=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp
                )
            except TypeError:
                # older client expects 'prompt' arg
                result = client.text_generation(
                    model=HF_API_MODEL,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp
                )

            # Normalize result shapes
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    return first.get("generated_text") or first.get("text") or str(first)
                return str(first)
            if isinstance(result, dict):
                return result.get("generated_text") or result.get("text") or str(result)
            return str(result)

        except Exception as e:
            # Capture any Exception (StopIteration, provider routing error, auth, etc.)
            client_exc_info = repr(e)

    # If we reach here, either client not available or client call failed -> raw HTTP fallback
    ok, status, body_text, headers = _http_post_hf(prompt, max_new_tokens=max_new_tokens, temp=temp, timeout=timeout)
    if ok:
        # Try parsing JSON -> list/dict
        try:
            data = requests.models.json.loads(body_text) if isinstance(body_text, str) else body_text
        except Exception:
            return str(body_text)
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                return item.get("generated_text") or item.get("text") or str(item)
        return str(data)

    # Not ok -> return diagnostics including client exception info
    body_preview = (body_text[:2000] if body_text else "")
    return (None, status, body_preview, headers, client_exc_info)


def call_hf_inference_with_retry(prompt: str, tries=2, backoff=1.0, **kwargs):
    """
    Calls inference with retries. Does not retry on 4xx client errors.
    Returns either success string or failure tuple from call_hf_inference_once.
    """
    last_resp = None
    for attempt in range(1, tries + 1):
        out = call_hf_inference_once(prompt, **kwargs)
        if isinstance(out, tuple) and out[0] is None:
            # out = (None, status, body_preview, headers, client_exc_info)
            _, status, body_preview, headers, client_exc_info = out
            last_resp = out
            # If a client error (4xx) -> stop retrying
            if status and 400 <= status < 500:
                return out
            if attempt < tries:
                time.sleep(backoff * attempt)
                continue
            return out
        else:
            return out
    return last_resp


def rewrite_text(text: str, tone: str = "Neutral"):
    """
    Trims, chunks, calls HF with retry, concatenates results.
    If HF fails for a chunk, fallback to original chunk.
    Returns (rewritten_full_text, hf_problem_detected_flag, hf_diagnostics_list)
    diagnostics: list of {"part","status","body_preview","headers","fallback_info"}
    """
    diagnostics = []
    text = safe_trim(text)
    parts = chunk_text_by_chars(text, max_chars=DEFAULT_CHUNK_MAX)
    outputs = []
    hf_any_success = False

    for idx, p in enumerate(parts):
        prompt = build_prompt_chunk(p, tone)
        out = call_hf_inference_with_retry(prompt, max_new_tokens=300, temp=0.5, timeout=120, tries=2, backoff=1.0)
        if isinstance(out, tuple) and out[0] is None:
            # out = (None, status, body_preview, headers, client_exc_info)
            _, status, body_preview, headers, client_exc_info = out
            diagnostics.append({
                "part": idx,
                "status": status,
                "body_preview": body_preview,
                "headers": headers,
                "fallback_info": client_exc_info
            })
            outputs.append(p)  # fallback to original chunk
        else:
            outputs.append(out.strip())
            hf_any_success = True

    rewritten_full = "\n\n".join(outputs)
    return rewritten_full, (not hf_any_success), diagnostics
