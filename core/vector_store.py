"""
core/vector_store.py — MongoDB Atlas Vector Store
Embeddings: Jina AI API (free, 1M tokens/month, no torch needed)
Fallback:   sentence-transformers CPU (if JINA_API_KEY not set)

Jina AI setup (free, no CC):
  1. https://jina.ai → Sign up → get API key
  2. Add JINA_API_KEY to .env / Render env vars
  3. Model: jina-embeddings-v3 → 1024 dims
"""

import os
import time
import hashlib
import logging
import requests
from typing import List, Dict, Any, Optional

from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

for _lib in ("httpx","httpcore","urllib3","sentence_transformers",
             "transformers","huggingface_hub","filelock"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MONGODB_URI     = os.getenv("MONGODB_URI",   "")
DATABASE_NAME   = os.getenv("DB_NAME",       "knowledge_base")
COLLECTION_NAME = os.getenv("COLLECTION",    "docs")
INDEX_NAME      = os.getenv("INDEX_NAME",    "vector_idx")

# Jina AI (recommended on Render — no local model needed)
JINA_API_KEY    = os.getenv("JINA_API_KEY",  "")
JINA_MODEL      = os.getenv("JINA_MODEL",    "jina-embeddings-v3")
JINA_DIMS       = 1024

# Local fallback (used when JINA_API_KEY not set — needs sentence-transformers)
LOCAL_MODEL     = os.getenv("EMBED_MODEL",   "BAAI/bge-small-en-v1.5")
LOCAL_DIMS      = 384

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",    "450"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "80"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.68"))
SEARCH_LIMIT    = int(os.getenv("SEARCH_LIMIT",   "5"))

# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING — Jina AI API (zero local dependencies)
# ════════════════════════════════════════════════════════════════════════════

def _jina_embed(texts: List[str]) -> List[List[float]]:
    """Call Jina AI embeddings API — free 1M tokens/month."""
    resp = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={"model": JINA_MODEL, "input": texts},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()["data"]
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING — Local fallback (sentence-transformers, CPU only)
# Only used when JINA_API_KEY is not set
# ════════════════════════════════════════════════════════════════════════════

_local_embedder = None
_local_dims     = None


def _get_local():
    global _local_embedder, _local_dims
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading local model: {LOCAL_MODEL} ...")
        _local_embedder = SentenceTransformer(LOCAL_MODEL)
        _local_dims     = _local_embedder.get_sentence_embedding_dimension()
        log.info(f"Local model ready — dim={_local_dims}")
    return _local_embedder


def _local_embed(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    emb = _get_local()
    out = []
    for i in range(0, len(texts), batch_size):
        vecs = emb.encode(texts[i:i+batch_size],
                          normalize_embeddings=True,
                          show_progress_bar=False)
        out.extend(vecs.tolist())
    return out


# ════════════════════════════════════════════════════════════════════════════
# UNIFIED EMBEDDING API
# ════════════════════════════════════════════════════════════════════════════

def _use_jina() -> bool:
    return bool(JINA_API_KEY)


def embed_batch(texts: List[str]) -> List[List[float]]:
    if _use_jina():
        # Jina has a 2048 input limit per call — batch if needed
        results = []
        for i in range(0, len(texts), 128):
            results.extend(_jina_embed(texts[i:i+128]))
        return results
    return _local_embed(texts)


def embed_one(text: str) -> List[float]:
    return embed_batch([text])[0]


def get_embedding_dims() -> int:
    return JINA_DIMS if _use_jina() else (
        _local_dims if _local_dims else LOCAL_DIMS
    )


def warm_up_embedder() -> None:
    if _use_jina():
        log.info(f"Embedding: Jina AI ({JINA_MODEL}, {JINA_DIMS}d) — no local model needed ✓")
    else:
        log.info(f"Embedding: local {LOCAL_MODEL} (set JINA_API_KEY for cloud-friendly embeddings)")
        _get_local()


# ════════════════════════════════════════════════════════════════════════════
# MONGODB
# ════════════════════════════════════════════════════════════════════════════

_client     = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI not set")
        _client     = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8_000)
        _client.admin.command("ping")
        _collection = _client[DATABASE_NAME][COLLECTION_NAME]
        log.info("MongoDB Atlas connected ✓")
    return _collection


def close_connection() -> None:
    global _client
    if _client:
        _client.close()
        _client = None


# ════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or not text.strip():
        return []
    overlap = min(overlap, chunk_size - 1)
    step    = chunk_size - overlap
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# INDEX
# ════════════════════════════════════════════════════════════════════════════

def ensure_index(timeout: int = 120) -> None:
    col  = get_collection()
    dims = get_embedding_dims()
    if any(idx["name"] == INDEX_NAME for idx in col.list_search_indexes()):
        log.info(f"Index '{INDEX_NAME}' exists ✓")
        return
    log.info(f"Creating vector index '{INDEX_NAME}' (dims={dims}) ...")
    col.create_search_index(SearchIndexModel(
        definition={"fields": [{
            "type": "vector", "path": "embedding",
            "numDimensions": dims, "similarity": "cosine",
        }]},
        name=INDEX_NAME, type="vectorSearch",
    ))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            idxs = list(col.list_search_indexes(INDEX_NAME))
            if idxs and idxs[0].get("queryable"):
                log.info("Index ready ✓")
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"Index not queryable after {timeout}s")


# ════════════════════════════════════════════════════════════════════════════
# INGESTION
# ════════════════════════════════════════════════════════════════════════════

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def insert_documents(documents: List[Dict[str, Any]],
                     skip_duplicates: bool = True,
                     user_id: Optional[str] = None) -> int:
    col = get_collection()
    txts, metas, hashes = [], [], []
    for doc in documents:
        content = (doc.get("content") or "").strip()
        if not content:
            continue
        h     = _sha256(content)
        query = {"content_hash": h, **({"user_id": user_id} if user_id else {})}
        if skip_duplicates and col.count_documents(query, limit=1):
            continue
        for chunk in chunk_text(content):
            meta = dict(doc.get("metadata", {}))
            if user_id:
                meta["user_id"] = user_id
            txts.append(chunk); metas.append(meta); hashes.append(h)

    if not txts:
        return 0
    log.info(f"Embedding {len(txts)} chunks ...")
    embeddings = embed_batch(txts)
    col.insert_many([
        {"text": txts[i], "embedding": embeddings[i],
         "metadata": metas[i], "content_hash": hashes[i],
         **({"user_id": user_id} if user_id else {})}
        for i in range(len(txts))
    ], ordered=False)
    log.info(f"Inserted {len(txts)} chunks ✓")
    return len(txts)


# ════════════════════════════════════════════════════════════════════════════
# SEARCH
# ════════════════════════════════════════════════════════════════════════════

def semantic_search(
    query: str, limit: int = SEARCH_LIMIT,
    score_threshold: float = SCORE_THRESHOLD,
    metadata_filter: Optional[Dict] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    col  = get_collection()
    qvec = embed_one(query)
    pipeline = [
        {"$vectorSearch": {
            "index": INDEX_NAME, "path": "embedding",
            "queryVector": qvec, "numCandidates": max(limit*10, 50),
            "limit": limit * 2,
        }},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$match": {"score": {"$gte": score_threshold}}},
    ]
    if user_id:
        pipeline.append({"$match": {"user_id": user_id}})
    if metadata_filter:
        pipeline.append({"$match": metadata_filter})
    pipeline += [
        {"$project": {"text": 1, "score": 1, "metadata": 1, "_id": 0}},
        {"$sort": {"score": -1}},
        {"$limit": limit},
    ]
    try:
        return list(col.aggregate(pipeline))
    except OperationFailure as exc:
        log.error(f"Search failed: {exc}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# STATS
# ════════════════════════════════════════════════════════════════════════════

def get_stats(user_id: Optional[str] = None) -> Dict[str, Any]:
    col   = get_collection()
    query = {"user_id": user_id} if user_id else {}
    return {
        "total_chunks":   col.count_documents(query),
        "unique_sources": len(col.distinct("metadata.source", query)),
        "sources":        col.distinct("metadata.source", query)[:20],
        "embed_model":    JINA_MODEL if _use_jina() else LOCAL_MODEL,
        "embed_dims":     get_embedding_dims(),
    }