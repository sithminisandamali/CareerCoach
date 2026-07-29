"""
llm_clients.py
Thin wrappers around Groq and OpenRouter chat-completions endpoints.
Both are OpenAI-compatible, so we use plain `requests` (no heavy SDK needed).

Model selection strategy (see README for full justification table):
    - Groq  llama-3.1-8b-instant      -> cheap/fast routing & classification
    - Groq  llama-3.3-70b-versatile   -> mid-tier, used for retrieval re-ranking
    - OpenRouter (free-tier model, verified working) -> deep reasoning: critique & final synthesis
"""

import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_secret(key: str) -> str:
    """Works both locally (env var) and on Streamlit Cloud (st.secrets)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, "")


def call_groq(model: str, messages: list, temperature: float = 0.3, max_tokens: int = 800) -> str:
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        return "[ERROR] GROQ_API_KEY not set. Add it to .streamlit/secrets.toml or env vars."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR calling Groq model {model}] {e}"


def call_openrouter(model: str, messages: list, temperature: float = 0.4, max_tokens: int = 1000) -> str:
    api_key = _get_secret("OPENROUTER_API_KEY")
    if not api_key:
        return "[ERROR] OPENROUTER_API_KEY not set. Add it to .streamlit/secrets.toml or env vars."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        content = message.get("content")
        if not content:
            # Some reasoning models leave content empty and put the real
            # answer in `reasoning` instead — fall back to that.
            content = message.get("reasoning")
        return content or "[ERROR] Model returned an empty response. Try rephrasing or increasing max_tokens."
    except requests.exceptions.HTTPError as e:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            detail = resp.text
        return f"[ERROR calling OpenRouter model {model}] {e} — {detail}"
    except Exception as e:
        return f"[ERROR calling OpenRouter model {model}] {e}"


def call_openrouter_with_fallback(messages: list, **kwargs) -> str:
    """Tries each candidate free model in order until one works."""
    last_error = ""
    for model in MODEL_DEEP_CANDIDATES:
        result = call_openrouter(model, messages, **kwargs)
        if not result.startswith("[ERROR"):
            return result
        last_error = result
    return last_error


# Central place to change models -> makes the "model selection strategy"
# easy to point to in the viva.
MODEL_ROUTER = "llama-3.1-8b-instant"          # Groq: routing/classification
MODEL_RERANK = "llama-3.3-70b-versatile"       # Groq: retrieval re-ranking

# OpenRouter free-tier models for critique/synthesis.
# Confirmed live via /api/v1/models on 2026-07-29 — free model IDs change
# often, so re-check https://openrouter.ai/models?order=top-weekly if this breaks again.
MODEL_DEEP_CANDIDATES = [
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
]
MODEL_DEEP = MODEL_DEEP_CANDIDATES[0]  # primary pick, used if you call MODEL_DEEP directly anywhere