"""
core/reranker.py — CrossEncoder Re-Ranker
NOTE: Disabled on cloud (Render) to save RAM — torch CPU still 250MB.
Set RERANK_ENABLED=false in Render env vars.
Local development: set RERANK_ENABLED=true
"""

import os
import logging
import math
from typing import List, Dict, Any

log = logging.getLogger(__name__)

RERANK_ENABLED    = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL      = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K      = int(os.getenv("RERANK_TOP_K",       "3"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        log.info(f"Loading re-ranker: {RERANK_MODEL} ...")
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        log.info("Re-ranker ready ✓")
    return _reranker


def warm_up_reranker() -> None:
    if RERANK_ENABLED:
        get_reranker()
    else:
        log.info("Re-ranker disabled (RERANK_ENABLED=false) — skipping load")


def rerank(query: str, chunks: List[Dict[str, Any]],
           top_k: int = RERANK_TOP_K) -> List[Dict[str, Any]]:
    """Re-rank using CrossEncoder if enabled, else return top-k by vector score."""
    if not chunks:
        return []
    if not RERANK_ENABLED:
        # Simple fallback: just return top-k sorted by existing vector score
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    try:
        model  = get_reranker()
        scores = model.predict(
            [(query, c["text"]) for c in chunks], show_progress_bar=False
        )
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in ranked[:top_k]:
            chunk = dict(chunk)
            chunk["vector_score"]  = chunk.get("score", 0.0)
            chunk["rerank_score"]  = float(score)
            chunk["score"]         = round(1 / (1 + math.exp(-float(score))), 4)
            result.append(chunk)
        return result
    except Exception as exc:
        log.warning(f"Re-ranker failed, using vector order: {exc}")
        return chunks[:top_k]


def rerank_and_explain(query: str, chunks: List[Dict[str, Any]],
                       top_k: int = RERANK_TOP_K) -> Dict[str, Any]:
    ranked     = rerank(query, chunks, top_k=top_k)
    if not ranked:
        return {"chunks": [], "best_score": 0.0, "score_gap": 0.0, "confidence": "Low"}
    scores     = [c["score"] for c in ranked]
    best_score = scores[0]
    score_gap  = round(scores[0] - scores[1], 3) if len(scores) > 1 else 1.0
    confidence = "High" if best_score >= 0.80 else "Medium" if best_score >= 0.60 else "Low"
    return {
        "chunks": ranked, "best_score": best_score,
        "score_gap": score_gap, "confidence": confidence,
    }