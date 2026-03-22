"""
core/rag.py — RAG Pipeline with Free LLM Backends
Supports: Docker Model Runner · Ollama (local) · Groq (free cloud) · HuggingFace (free API)
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Generator, Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
LLM_BACKEND  = os.getenv("LLM_BACKEND",  "docker")       # docker | ollama | groq | huggingface

# Docker Model Runner (OpenAI-compatible API built into Docker Desktop)
DOCKER_URL   = os.getenv("DOCKER_URL",   "http://localhost:12434")
DOCKER_MODEL = os.getenv("DOCKER_MODEL", "llama3.2")

# Ollama (standalone local install)
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Groq (free cloud)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.1-8b-instant")

# HuggingFace Inference API (free tier)
HF_API_KEY   = os.getenv("HF_API_KEY",   "")
HF_MODEL     = os.getenv("HF_MODEL",     "mistralai/Mistral-7B-Instruct-v0.3")

MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "512"))

# ── System Prompt ─────────────────────────────────────────────────────────────
# ── Recommendation mode prompt ──────────────────────────────────────────────
RECOMMENDATION_PROMPT = """You are a smart recommendation engine. Based on the provided context, give the BEST recommendation for the user's query.

Return ONLY a valid JSON object in this exact format (no markdown, no extra text):
{
  "best_match": "specific name, title, or option that best fits",
  "why": "2-3 sentence explanation of exactly why this is the best match for the query",
  "confidence": 85,
  "alternatives": ["second best option", "third best option"],
  "tips": "one practical tip for getting the most value from this recommendation",
  "warning": "important caveat if any, or null"
}"""

# ── Standard Q&A prompt (used when recommendation mode is off) ───────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided context.\n"
    "- Answer concisely and clearly.\n"
    "- If the context lacks enough information, say so honestly.\n"
    "- Do NOT invent facts outside the given context.\n"
    "- Cite relevant source names when possible."
)

def _build_messages(query: str, chunks: List[Dict],
                    mode: str = "qa") -> List[Dict]:
    """Build OpenAI-style messages list with context injected."""
    context_parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("metadata", {}).get("source", "Unknown")
        context_parts.append(f"[Source {i}: {source}]\n{c['text']}")
    context = "\n\n".join(context_parts)
    system  = RECOMMENDATION_PROMPT if mode == "recommend" else SYSTEM_PROMPT
    user_msg = f"=== CONTEXT ===\n{context}\n\n=== QUERY ===\n{query}"
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]


# ════════════════════════════════════════════════════════════════════════════
# BACKEND: DOCKER MODEL RUNNER
# Built into Docker Desktop 4.40+ — no extra install needed.
# Enable: Docker Desktop → Settings → Features in development → Model Runner
# Your models: ai/llama3.2:latest (LLM) · ai/nomic-embed-text-v1.5 (embeddings)
# API base: http://localhost:12434/engines/llama.cpp/v1  (OpenAI-compatible)
# ════════════════════════════════════════════════════════════════════════════

def _docker_generate(messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
    url     = f"{DOCKER_URL}/engines/llama.cpp/v1/chat/completions"
    payload = {
        "model":       DOCKER_MODEL,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": 0.3,
        "stream":      stream,
    }
    try:
        resp = requests.post(url, json=payload, stream=stream, timeout=300)
        resp.raise_for_status()

        if stream:
            for line in resp.iter_lines():
                if not line:
                    continue
                text  = line.decode("utf-8") if isinstance(line, bytes) else line
                if not text.startswith("data: "):
                    continue
                chunk = text[6:]
                if chunk.strip() == "[DONE]":
                    break
                data  = json.loads(chunk)
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
        else:
            yield resp.json()["choices"][0]["message"]["content"]

    except requests.exceptions.ConnectionError:
        yield (
            "❌ Docker Model Runner not reachable at port 12434.\n"
            "Make sure Docker Desktop is running and Model Runner is enabled:\n"
            "Docker Desktop → Settings → Features in development → Enable Docker Model Runner"
        )
    except Exception as exc:
        yield f"❌ Docker error: {exc}"


def docker_list_models() -> List[str]:
    """Return models available in Docker Model Runner."""
    try:
        resp = requests.get(
            f"{DOCKER_URL}/engines/llama.cpp/v1/models", timeout=5
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []



def docker_warmup() -> bool:
    """
    Send a tiny request to load the model into memory before first real query.
    Call this on startup so the user never sees a timeout.
    """
    try:
        resp = requests.post(
            f"{DOCKER_URL}/engines/llama.cpp/v1/chat/completions",
            json={
                "model":      DOCKER_MODEL,
                "messages":   [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream":     False,
            },
            timeout=300,
        )
        return resp.status_code == 200
    except Exception as exc:
        log.warning(f"Docker LLM warmup failed (will retry on first query): {exc}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# BACKEND: OLLAMA
# ════════════════════════════════════════════════════════════════════════════

def _ollama_generate(messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
    url     = f"{OLLAMA_URL}/v1/chat/completions"
    payload = {"model": OLLAMA_MODEL, "messages": messages,
                "max_tokens": MAX_TOKENS, "temperature": 0.3, "stream": stream}
    try:
        resp = requests.post(url, json=payload, stream=stream, timeout=300)
        resp.raise_for_status()
        if stream:
            for line in resp.iter_lines():
                if not line:
                    continue
                text  = line.decode("utf-8") if isinstance(line, bytes) else line
                if not text.startswith("data: "):
                    continue
                chunk = text[6:]
                if chunk.strip() == "[DONE]":
                    break
                data  = json.loads(chunk)
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
        else:
            yield resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        yield "❌ Ollama is not running. Start it with: `ollama serve`"
    except Exception as exc:
        yield f"❌ Ollama error: {exc}"


def ollama_list_models() -> List[str]:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════════════════
# BACKEND: GROQ
# ════════════════════════════════════════════════════════════════════════════

def _groq_generate(messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
    if not GROQ_API_KEY:
        yield "❌ GROQ_API_KEY not set — get a free key at console.groq.com"
        return
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages,
                "max_tokens": MAX_TOKENS, "temperature": 0.3, "stream": stream}
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers=headers, json=payload, stream=stream, timeout=60)
        resp.raise_for_status()
        if stream:
            for line in resp.iter_lines():
                if not line:
                    continue
                text  = line.decode("utf-8") if isinstance(line, bytes) else line
                if not text.startswith("data: "):
                    continue
                chunk = text[6:]
                if chunk.strip() == "[DONE]":
                    break
                data  = json.loads(chunk)
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
        else:
            yield resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        yield f"❌ Groq error: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# BACKEND: HUGGINGFACE
# ════════════════════════════════════════════════════════════════════════════

def _hf_generate(messages: List[Dict]) -> Generator[str, None, None]:
    if not HF_API_KEY:
        yield "❌ HF_API_KEY not set — get a free token at huggingface.co/settings/tokens"
        return
    prompt = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages
    ) + "\nAssistant:"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {
        "max_new_tokens": MAX_TOKENS, "temperature": 0.3, "return_full_text": False}}
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        yield result[0].get("generated_text", "") if isinstance(result, list) and result else str(result)
    except Exception as exc:
        yield f"❌ HuggingFace error: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC RAG INTERFACE
# ════════════════════════════════════════════════════════════════════════════

def generate_answer(
    query:   str,
    chunks:  List[Dict],
    backend: Optional[str] = None,
    stream:  bool = True,
    mode:    str  = "qa",          # "qa" | "recommend"
) -> Generator[str, None, None]:
    if not chunks:
        yield "I couldn't find relevant information. Try uploading more documents."
        return

    messages = _build_messages(query, chunks, mode=mode)
    backend  = (backend or LLM_BACKEND).lower()
    log.info(f"RAG → backend={backend}, chunks={len(chunks)}")

    if   backend == "docker":      yield from _docker_generate(messages, stream=stream)
    elif backend == "ollama":      yield from _ollama_generate(messages, stream=stream)
    elif backend == "groq":        yield from _groq_generate(messages, stream=stream)
    elif backend == "huggingface": yield from _hf_generate(messages)
    else:
        yield f"❌ Unknown backend '{backend}'. Choose: docker | ollama | groq | huggingface"


def get_backend_info() -> Dict[str, Any]:
    return {
        "current": LLM_BACKEND,
        "docker": {
            "url":    DOCKER_URL,
            "model":  DOCKER_MODEL,
            "models": docker_list_models(),
        },
        "ollama": {
            "url":    OLLAMA_URL,
            "model":  OLLAMA_MODEL,
            "models": ollama_list_models(),
        },
        "groq":        {"model": GROQ_MODEL,   "configured": bool(GROQ_API_KEY)},
        "huggingface": {"model": HF_MODEL,     "configured": bool(HF_API_KEY)},
    }