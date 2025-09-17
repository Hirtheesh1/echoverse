# backend/rewrite.py
"""
Gemini-backed rewrite module.

Behavior:
 - Tries to use google-genai SDK when available (preferred).
 - Falls back to REST POST to the Gemini generateContent endpoint if SDK isn't present.
 - Exposes env vars: GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT, GEMINI_TRIES, GEMINI_BACKOFF, GEMINI_THINKING_BUDGET.
 - Public: rewrite_text(text: str, tone: str="Neutral") -> (rewritten_full_text, inference_problem_detected_flag, diagnostics_list)
"""
from __future__ import annotations

import os
import time
import json
import typing as t
from typing import Any, Dict, Optional, Tuple, Union

from dotenv import load_dotenv, find_dotenv
from .utils import chunk_text_by_chars, safe_trim

# Load .env (if present)
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv(".env")

# Try to import google-genai SDK
try:
    from google import genai  # type: ignore

    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    genai = None  # type: ignore
    GOOGLE_GENAI_AVAILABLE = False

# Config via env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # used by REST fallback
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "240"))
GEMINI_TRIES = int(os.getenv("GEMINI_TRIES", "3"))
GEMINI_BACKOFF = float(os.getenv("GEMINI_BACKOFF", "2.0"))
# Thinking budget: if you want to disable thinking, set 0
GEMINI_THINKING_BUDGET = None
_env_think = os.getenv("GEMINI_THINKING_BUDGET")
if _env_think is not None:
    try:
        GEMINI_THINKING_BUDGET = int(_env_think)
    except Exception:
        GEMINI_THINKING_BUDGET = None

# REST endpoint for generateContent (per docs)
GEMINI_REST_BASE = os.getenv("GEMINI_REST_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_REST_URL_TEMPLATE = GEMINI_REST_BASE + "/models/{model}:generateContent"

# Diagnostics helper
def _short_preview(obj: Any, length: int = 2000) -> str:
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    except Exception:
        s = str(obj)
    if len(s) > length:
        return s[:length] + "..."
    return s

# Prompt builder
def build_prompt_chunk(text: str, tone: str):
    instr = (
        "Rewrite the following text to improve clarity, structure, and flow while preserving "
        "all facts, numbers, and the original meaning. Be concise and do not add new information.\n\n"
        f"Tone: {tone}\n\n"
        "Original:\n"
        f"{text}\n\n"
        "Rewritten:"
    )
    return instr

# ----------------- SDK path (preferred) -----------------
def _genai_client_singleton():
    if not GOOGLE_GENAI_AVAILABLE:
        return None
    if not hasattr(_genai_client_singleton, "client"):
        if GEMINI_API_KEY:
            os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)
        _genai_client_singleton.client = genai.Client()
    return getattr(_genai_client_singleton, "client")

def _call_gemini_sdk_once(prompt: str, model: Optional[str] = None, timeout: int = GEMINI_TIMEOUT, thinking_budget: Optional[int] = GEMINI_THINKING_BUDGET):
    model = model or GEMINI_MODEL
    client = _genai_client_singleton()
    if client is None:
        return None

    try:
        config = None
        if thinking_budget is not None:
            try:
                from google.genai import types  # type: ignore
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=int(thinking_budget))
                )
            except Exception:
                config = {"thinkingConfig": {"thinkingBudget": int(thinking_budget)}}

        if config is not None:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
        else:
            resp = client.models.generate_content(model=model, contents=prompt)

        try:
            text = resp.text if isinstance(resp.text, str) else str(resp.text)
        except Exception:
            try:
                text = resp.text()
            except Exception:
                text = str(resp)
        return text

    except Exception as e:
        return (None, None, _short_preview(repr(e)), {}, repr(e))

# ----------------- REST fallback -----------------
import requests

def _post_gemini_rest(payload: Dict[str, Any], timeout: int = GEMINI_TIMEOUT) -> Tuple[bool, Optional[int], Any, Dict[str, str]]:
    model = payload.get("model", GEMINI_MODEL)
    url = GEMINI_REST_URL_TEMPLATE.format(model=model)
    headers = {"Content-Type": "application/json"}
    if GEMINI_API_KEY:
        headers["x-goog-api-key"] = GEMINI_API_KEY

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return resp.ok, status, body, dict(resp.headers)
    except Exception as e:
        return False, None, f"exception:{e}", {}

def _parse_gemini_rest_result(body: Any) -> str:
    try:
        if isinstance(body, dict):
            cands = body.get("candidates") or []
            if cands:
                first = cands[0]
                content = first.get("content") or {}
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                    if parts and isinstance(parts, list):
                        p0 = parts[0]
                        if isinstance(p0, dict) and "text" in p0:
                            return p0["text"]
                        if isinstance(p0, str):
                            return p0
            if "output" in body and isinstance(body["output"], str):
                return body["output"]
        return str(body)
    except Exception:
        return str(body)

def _call_gemini_rest_once(prompt: str, model: Optional[str] = None, timeout: int = GEMINI_TIMEOUT, thinking_budget: Optional[int] = GEMINI_THINKING_BUDGET):
    model = model or GEMINI_MODEL
    contents = [{"parts": [{"text": prompt}]}]
    payload: Dict[str, Any] = {"model": model, "contents": contents}
    if thinking_budget is not None:
        payload["generationConfig"] = {"thinkingConfig": {"thinkingBudget": int(thinking_budget)}}

    ok, status, body, headers = _post_gemini_rest(payload, timeout=timeout)
    if ok:
        try:
            return _parse_gemini_rest_result(body)
        except Exception:
            return str(body)
    body_preview = _short_preview(body)
    return (None, status, body_preview, headers, None)

# ----------------- Unified single-attempt caller -----------------
def call_gemini_once(prompt: str, model: Optional[str] = None, timeout: int = GEMINI_TIMEOUT, thinking_budget: Optional[int] = GEMINI_THINKING_BUDGET):
    if GOOGLE_GENAI_AVAILABLE:
        out = _call_gemini_sdk_once(prompt, model=model, timeout=timeout, thinking_budget=thinking_budget)
        if out is None:
            pass
        elif isinstance(out, tuple) and out[0] is None:
            return out
        else:
            return str(out)
    return _call_gemini_rest_once(prompt, model=model, timeout=timeout, thinking_budget=thinking_budget)

# ----------------- Retry wrapper -----------------
def call_gemini_with_retry(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = GEMINI_TIMEOUT,
    tries: int = GEMINI_TRIES,
    backoff: float = GEMINI_BACKOFF,
    thinking_budget: Optional[int] = GEMINI_THINKING_BUDGET,
):
    last_resp = None
    for attempt in range(1, tries + 1):
        out = call_gemini_once(prompt, model=model, timeout=timeout, thinking_budget=thinking_budget)
        if isinstance(out, tuple) and out[0] is None:
            _, status, body_preview, headers, client_exc = out
            last_resp = out
            if status == 429 and headers:
                ra = headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else backoff * attempt
                except Exception:
                    wait = backoff * attempt
                time.sleep(wait)
                continue
            if status and 400 <= status < 500:
                return out
            if attempt < tries:
                time.sleep(backoff * attempt)
                continue
            return out
        else:
            return out
    return last_resp

# ----------------- Top-level rewrite_text -----------------
DEFAULT_CHUNK_MAX = int(os.getenv("DEFAULT_CHUNK_MAX", "3000"))

def rewrite_text(text: str, tone: str = "Neutral"):
    diagnostics: t.List[Dict[str, Any]] = []
    text = safe_trim(text)
    parts = chunk_text_by_chars(text, max_chars=DEFAULT_CHUNK_MAX)
    outputs: t.List[str] = []
    any_success = False

    for idx, p in enumerate(parts):
        prompt = build_prompt_chunk(p, tone)
        out = call_gemini_with_retry(
            prompt,
            model=os.getenv("GEMINI_MODEL", GEMINI_MODEL),
            timeout=int(os.getenv("GEMINI_TIMEOUT", GEMINI_TIMEOUT)),
            tries=int(os.getenv("GEMINI_TRIES", GEMINI_TRIES)),
            backoff=float(os.getenv("GEMINI_BACKOFF", GEMINI_BACKOFF)),
            thinking_budget=(int(os.getenv("GEMINI_THINKING_BUDGET")) if os.getenv("GEMINI_THINKING_BUDGET") else GEMINI_THINKING_BUDGET),
        )

        if isinstance(out, tuple) and out[0] is None:
            _, status, body_preview, headers, client_exc_info = out
            diagnostics.append(
                {
                    "part": idx,
                    "status": status,
                    "body_preview": body_preview,
                    "headers": headers,
                    "fallback_info": client_exc_info,
                }
            )
            outputs.append(p)
        else:
            outputs.append(out.strip())
            any_success = True

    rewritten_full = "\n\n".join(outputs)
    return rewritten_full, (not any_success), diagnostics

if __name__ == "__main__":
    sample = "This is a sample paragraph with a number 2025. Please rewrite it clearly."
    print("SDK available:", GOOGLE_GENAI_AVAILABLE)
    print("GEMINI_MODEL:", GEMINI_MODEL)
    out, failed, diags = rewrite_text(sample, tone="Neutral")
    print("Failed:", failed)
    print("Diags:", diags)
    print("Output:", out)
