"""
core/rag.py — RAG Pipeline
Backends: Groq (recommended, free) · HuggingFace (free)
Docker and Ollama removed — not available on cloud hosting.
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Generator, Optional

log = logging.getLogger(__name__)

LLM_BACKEND  = os.getenv("LLM_BACKEND",  "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.1-8b-instant")

HF_API_KEY   = os.getenv("HF_API_KEY",   "")
HF_MODEL     = os.getenv("HF_MODEL",     "mistralai/Mistral-7B-Instruct-v0.3")

MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "512"))

# ── Prompts ───────────────────────────────────────────────────────────────────
RECOMMENDATION_PROMPT = """You are a smart recommendation engine. Based on the provided context, give the BEST recommendation for the user's query.

Return ONLY a valid JSON object in this exact format (no markdown, no extra text):
{
  "best_match": "specific name, title, or option that best fits",
  "why": "2-3 sentence explanation of exactly why this is the best match",
  "confidence": 85,
  "alternatives": ["second best option", "third best option"],
  "tips": "one practical tip for getting the most value",
  "warning": "important caveat if any, or null"
}"""

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided context.\n"
    "- Answer concisely and clearly.\n"
    "- If the context lacks enough information, say so honestly.\n"
    "- Do NOT invent facts outside the given context.\n"
    "- Cite relevant source names when possible."
)


def _build_messages(query: str, chunks: List[Dict],
                    mode: str = "qa") -> List[Dict]:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("metadata", {}).get("source", "Unknown")
        context_parts.append(f"[Source {i}: {source}]\n{c['text']}")
    context  = "\n\n".join(context_parts)
    system   = RECOMMENDATION_PROMPT if mode == "recommend" else SYSTEM_PROMPT
    user_msg = f"=== CONTEXT ===\n{context}\n\n=== QUERY ===\n{query}"
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]


# ════════════════════════════════════════════════════════════════════════════
# GROQ (free cloud — console.groq.com, no credit card)
# Free models: llama-3.1-8b-instant, llama3-70b-8192, mixtral-8x7b-32768
# ════════════════════════════════════════════════════════════════════════════

def _groq_generate(messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
    if not GROQ_API_KEY:
        yield "Groq API key not set. Add GROQ_API_KEY to your environment variables."
        return
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": 0.3,
        "stream":      stream,
    }
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, stream=stream, timeout=60,
        )
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
        yield f"Groq error: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE (free tier — huggingface.co/settings/tokens)
# ════════════════════════════════════════════════════════════════════════════

def _hf_generate(messages: List[Dict]) -> Generator[str, None, None]:
    if not HF_API_KEY:
        yield "HuggingFace API key not set. Add HF_API_KEY to your environment variables."
        return
    prompt = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages
    ) + "\nAssistant:"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   MAX_TOKENS,
            "temperature":      0.3,
            "return_full_text": False,
        },
    }
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers, json=payload, timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        yield result[0].get("generated_text", "") if isinstance(result, list) and result else str(result)
    except Exception as exc:
        yield f"HuggingFace error: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC RAG INTERFACE
# ════════════════════════════════════════════════════════════════════════════

def generate_answer(
    query:   str,
    chunks:  List[Dict],
    backend: Optional[str] = None,
    stream:  bool = True,
    mode:    str  = "qa",
) -> Generator[str, None, None]:
    if not chunks:
        yield "I couldn't find relevant information. Try uploading more documents."
        return

    messages = _build_messages(query, chunks, mode=mode)
    backend  = (backend or LLM_BACKEND).lower()
    log.info(f"RAG → backend={backend}, chunks={len(chunks)}, mode={mode}")

    if backend == "groq":
        yield from _groq_generate(messages, stream=stream)
    elif backend == "huggingface":
        yield from _hf_generate(messages)
    else:
        # Default to Groq
        yield from _groq_generate(messages, stream=stream)


def docker_warmup() -> bool:
    """No-op — Docker removed. Kept for import compatibility."""
    return True


def get_backend_info() -> Dict[str, Any]:
    return {
        "current": LLM_BACKEND,
        "groq": {
            "model":      GROQ_MODEL,
            "configured": bool(GROQ_API_KEY),
        },
        "huggingface": {
            "model":      HF_MODEL,
            "configured": bool(HF_API_KEY),
        },
    }