"""
core/feedback.py — User Feedback Loop
Stores 👍/👎 per query/chunk and uses it to boost/penalise future searches.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pymongo import DESCENDING

log = logging.getLogger(__name__)

FEEDBACK_COLLECTION = os.getenv("FEEDBACK_COLLECTION", "feedback")


def get_feedback_col():
    from core.vector_store import get_collection
    col = get_collection()
    return col.database[FEEDBACK_COLLECTION]


# ════════════════════════════════════════════════════════════════════════════
# SAVE FEEDBACK
# ════════════════════════════════════════════════════════════════════════════

def save_feedback(
    user_id:    str,
    query:      str,
    chunk_text: str,
    source:     str,
    rating:     int,          # +1 = thumbs up, -1 = thumbs down
    answer:     str = "",
    mode:       str = "qa",
) -> bool:
    try:
        col = get_feedback_col()
        col.update_one(
            {
                "user_id":    user_id,
                "query":      query,
                "chunk_text": chunk_text[:200],
            },
            {"$set": {
                "rating":     rating,
                "source":     source,
                "answer":     answer[:500],
                "mode":       mode,
                "updated_at": datetime.utcnow(),
                "user_id":    user_id,
                "query":      query,
                "chunk_text": chunk_text[:200],
            }},
            upsert=True,
        )
        log.info(f"Feedback saved: {'👍' if rating > 0 else '👎'} from {user_id}")
        return True
    except Exception as exc:
        log.error(f"Feedback save failed: {exc}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# APPLY FEEDBACK TO SEARCH RESULTS
# Re-rank chunks using stored feedback — boost liked, penalise disliked
# ════════════════════════════════════════════════════════════════════════════

def apply_feedback_boost(
    user_id: str,
    query:   str,
    chunks:  List[Dict[str, Any]],
    boost:   float = 0.08,     # how much to boost/penalise per rating
) -> List[Dict[str, Any]]:
    """
    Adjust chunk scores based on past feedback for similar queries.
    Liked chunks get a small score boost, disliked get penalised.
    """
    if not chunks or not user_id:
        return chunks

    try:
        col = get_feedback_col()
        feedback_docs = list(col.find(
            {"user_id": user_id},
            {"chunk_text": 1, "rating": 1, "_id": 0},
            limit=200,
        ))

        if not feedback_docs:
            return chunks

        # Build lookup: chunk_text_prefix → rating
        fb_map = {
            f["chunk_text"][:100]: f["rating"]
            for f in feedback_docs
        }

        for chunk in chunks:
            key    = chunk.get("text", "")[:100]
            rating = fb_map.get(key, 0)
            if rating != 0:
                chunk["score"] = min(1.0, max(0.0,
                    chunk.get("score", 0.5) + (boost * rating)
                ))
                chunk["feedback_applied"] = rating

        # Re-sort after boost
        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        return chunks

    except Exception as exc:
        log.warning(f"Feedback boost failed (returning original order): {exc}")
        return chunks


# ════════════════════════════════════════════════════════════════════════════
# FEEDBACK STATS (for dashboard)
# ════════════════════════════════════════════════════════════════════════════

def get_feedback_stats(user_id: str) -> Dict[str, Any]:
    try:
        col   = get_feedback_col()
        query = {"user_id": user_id}
        total = col.count_documents(query)
        likes = col.count_documents({**query, "rating": 1})
        dislikes = col.count_documents({**query, "rating": -1})

        # Most liked sources
        pipeline = [
            {"$match": {**query, "rating": 1}},
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort":  {"count": -1}},
            {"$limit": 5},
        ]
        top_sources = [
            {"source": r["_id"], "likes": r["count"]}
            for r in col.aggregate(pipeline)
        ]

        # Recent feedback
        recent = list(col.find(
            query,
            {"query": 1, "rating": 1, "source": 1, "updated_at": 1, "_id": 0},
            sort=[("updated_at", DESCENDING)],
            limit=10,
        ))
        for r in recent:
            if "updated_at" in r:
                r["updated_at"] = r["updated_at"].isoformat()

        return {
            "total":       total,
            "likes":       likes,
            "dislikes":    dislikes,
            "top_sources": top_sources,
            "recent":      recent,
            "satisfaction": round((likes / total * 100) if total else 0, 1),
        }
    except Exception as exc:
        log.error(f"Feedback stats error: {exc}")
        return {"total": 0, "likes": 0, "dislikes": 0,
                "top_sources": [], "recent": [], "satisfaction": 0}


def get_global_stats() -> Dict[str, Any]:
    """Admin-level stats across all users."""
    try:
        col   = get_feedback_col()
        total = col.count_documents({})
        likes = col.count_documents({"rating":  1})
        dislikes = col.count_documents({"rating": -1})
        return {
            "total": total, "likes": likes, "dislikes": dislikes,
            "satisfaction": round((likes / total * 100) if total else 0, 1),
        }
    except Exception as exc:
        log.error(f"Global feedback stats error: {exc}")
        return {}