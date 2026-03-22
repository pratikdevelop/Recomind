"""
core/reranker.py — Cross-Encoder Re-Ranker
Improves recommendation quality by re-scoring vector search candidates
using a more accurate (but slower) cross-encoder model.

Pipeline:
  vector search (fast, recall-focused, fetches ~20 candidates)
      ↓
  cross-encoder re-rank (accurate, precision-focused, keeps top-k)
      ↓
  LLM generates recommendation from best chunks only

Free model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, fast on CPU
  - Trained on MS MARCO passage ranking
  - Much better relevance than cosine similarity alone
"""

import os
import logging
from typing import List, Dict, Any

log = logging.getLogger(__name__)

RERANK_MODEL   = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_TOP_K   = int(os.getenv("RERANK_TOP_K", "3"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        # Silence noisy logs from sentence_transformers during load
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)

        from sentence_transformers import CrossEncoder
        log.info(f"Loading re-ranker model: {RERANK_MODEL} ...")
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        log.info("Re-ranker ready ✓")
    return _reranker


def warm_up_reranker() -> None:
    """Call on startup to pre-load the cross-encoder into memory."""
    if RERANK_ENABLED:
        get_reranker()


def rerank(query: str, chunks: List[Dict[str, Any]],
           top_k: int = RERANK_TOP_K) -> List[Dict[str, Any]]:
    """
    Re-score chunks using a cross-encoder and return the top-k.

    Parameters
    ----------
    query  : the user's query string
    chunks : raw results from semantic_search()
    top_k  : how many to keep after re-ranking

    Returns
    -------
    Sorted list (best first), each chunk has an added 'rerank_score' field.
    The original vector 'score' is preserved as 'vector_score'.
    """
    if not chunks:
        return []

    if not RERANK_ENABLED:
        return chunks[:top_k]

    try:
        model  = get_reranker()
        pairs  = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in ranked[:top_k]:
            chunk = dict(chunk)                         # don't mutate original
            chunk["vector_score"] = chunk.get("score", 0.0)
            chunk["rerank_score"] = float(score)
            # Use rerank score as the primary display score (normalised 0–1)
            # CrossEncoder scores are unbounded; sigmoid squishes to 0–1
            import math
            chunk["score"] = round(1 / (1 + math.exp(-float(score))), 4)
            results.append(chunk)

        log.info(
            f"Re-ranked {len(chunks)} → {len(results)} chunks "
            f"(top score: {results[0]['score']:.3f})"
        )
        return results

    except Exception as exc:
        log.warning(f"Re-ranker failed, falling back to vector order: {exc}")
        return chunks[:top_k]


def rerank_and_explain(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = RERANK_TOP_K,
) -> Dict[str, Any]:
    """
    Re-rank and return both the ranked chunks and a score breakdown
    useful for the recommendation card UI.
    """
    ranked = rerank(query, chunks, top_k=top_k)
    if not ranked:
        return {"chunks": [], "best_score": 0.0, "score_gap": 0.0, "confidence": "Low"}

    scores     = [c["score"] for c in ranked]
    best_score = scores[0]
    score_gap  = round(scores[0] - scores[1], 3) if len(scores) > 1 else 1.0

    # Confidence label based on best rerank score
    if best_score >= 0.80:
        confidence = "High"
    elif best_score >= 0.60:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "chunks":     ranked,
        "best_score": best_score,
        "score_gap":  score_gap,
        "confidence": confidence,
    }