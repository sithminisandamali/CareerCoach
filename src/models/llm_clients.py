"""
llm_clients.py
Thin wrappers around Groq and OpenRouter chat-completions endpoints.
Both are OpenAI-compatible, so we use plain `requests` (no heavy SDK needed).

Model selection strategy (see README for full justification table):
    - Groq  llama-3.1-8b-instant      -> cheap/fast routing & classification
    - Groq  llama-3.3-70b-versatile   -> mid-tier, used for retrieval re-ranking
    - OpenRouter anthropic/claude-3.5-sonnet (or any OpenRouter model id)
                                      -> deep reasoning: critique & final synthesis
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
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR calling OpenRouter model {model}] {e}"


# Central place to change models -> makes the "model selection strategy"
# easy to point to in the viva.
MODEL_ROUTER = "llama-3.1-8b-instant"          # Groq: routing/classification
MODEL_RERANK = "llama-3.3-70b-versatile"       # Groq: retrieval re-ranking
MODEL_DEEP = "meta-llama/llama-3.3-70b-instruct:free"  # OpenRouter: critique/synthesis (free tier, no credits needed)