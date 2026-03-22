"""
core/vector_store.py — MongoDB Atlas Vector Store
Embedding: sentence-transformers (local, no Docker needed)
Works on: local machine, Render, Railway, any cloud
"""

import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.operations import SearchIndexModel

for _lib in ("httpx","httpcore","urllib3","sentence_transformers",
             "transformers","huggingface_hub","filelock","hpack"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MONGODB_URI     = os.getenv("MONGODB_URI",     "")
DATABASE_NAME   = os.getenv("DB_NAME",         "knowledge_base")
COLLECTION_NAME = os.getenv("COLLECTION",      "docs")
INDEX_NAME      = os.getenv("INDEX_NAME",      "vector_idx")
LOCAL_EMBED_MODEL = os.getenv("EMBED_MODEL",   "BAAI/bge-small-en-v1.5")

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",    "450"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "80"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.68"))
SEARCH_LIMIT    = int(os.getenv("SEARCH_LIMIT",   "5"))

# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING — sentence-transformers (works everywhere)
# ════════════════════════════════════════════════════════════════════════════

_embedder  = None
_dims: Optional[int] = None


def _get_embedder():
    global _embedder, _dims
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model: {LOCAL_EMBED_MODEL} ...")
        _embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
        _dims     = _embedder.get_sentence_embedding_dimension()
        log.info(f"Embedder ready — dim={_dims}")
    return _embedder


def embed_one(text: str) -> List[float]:
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    emb = _get_embedder()
    out = []
    for i in range(0, len(texts), batch_size):
        vecs = emb.encode(texts[i:i+batch_size],
                          normalize_embeddings=True,
                          show_progress_bar=False)
        out.extend(vecs.tolist())
    return out


def get_embedding_dims() -> int:
    _get_embedder()
    return _dims


def warm_up_embedder() -> None:
    log.info(f"Embedding model: {LOCAL_EMBED_MODEL}")
    get_embedding_dims()
    log.info(f"Embedder ready — dim={_dims}")


# ════════════════════════════════════════════════════════════════════════════
# MONGODB CONNECTION
# ════════════════════════════════════════════════════════════════════════════

_client     = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI not set in environment")
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8_000)
        _client.admin.command("ping")
        db          = _client[DATABASE_NAME]
        _collection = db[COLLECTION_NAME]
        log.info("MongoDB Atlas connected ✓")
    return _collection


def close_connection() -> None:
    global _client
    if _client:
        _client.close()
        _client = None


# ════════════════════════════════════════════════════════════════════════════
# TEXT CHUNKING
# ════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or not text.strip():
        return []
    overlap = min(overlap, chunk_size - 1)
    step    = chunk_size - overlap
    chunks  = []
    start   = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# VECTOR INDEX
# ════════════════════════════════════════════════════════════════════════════

def ensure_index(timeout: int = 120) -> None:
    col  = get_collection()
    dims = get_embedding_dims()

    existing = list(col.list_search_indexes())
    if any(idx["name"] == INDEX_NAME for idx in existing):
        log.info(f"Index '{INDEX_NAME}' exists ✓")
        return

    log.info(f"Creating vector index '{INDEX_NAME}' (dims={dims}) ...")
    model = SearchIndexModel(
        definition={"fields": [{
            "type":          "vector",
            "path":          "embedding",
            "numDimensions": dims,
            "similarity":    "cosine",
        }]},
        name=INDEX_NAME,
        type="vectorSearch",
    )
    col.create_search_index(model=model)

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
# DOCUMENT INGESTION
# ════════════════════════════════════════════════════════════════════════════

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def insert_documents(documents: List[Dict[str, Any]],
                     skip_duplicates: bool = True,
                     user_id: Optional[str] = None) -> int:
    col          = get_collection()
    chunk_texts  = []
    chunk_metas  = []
    chunk_hashes = []

    for doc in documents:
        content = (doc.get("content") or "").strip()
        if not content:
            continue
        h     = _sha256(content)
        query = {"content_hash": h}
        if user_id:
            query["user_id"] = user_id
        if skip_duplicates and col.count_documents(query, limit=1):
            log.info(f"  Duplicate skipped: {doc.get('metadata',{}).get('source','?')}")
            continue
        for chunk in chunk_text(content):
            meta = dict(doc.get("metadata", {}))
            if user_id:
                meta["user_id"] = user_id
            chunk_texts.append(chunk)
            chunk_metas.append(meta)
            chunk_hashes.append(h)

    if not chunk_texts:
        return 0

    log.info(f"Embedding {len(chunk_texts)} chunks ...")
    embeddings = embed_batch(chunk_texts)
    records    = [
        {
            "text":         chunk_texts[i],
            "embedding":    embeddings[i],
            "metadata":     chunk_metas[i],
            "content_hash": chunk_hashes[i],
            **({"user_id": user_id} if user_id else {}),
        }
        for i in range(len(chunk_texts))
    ]
    col.insert_many(records, ordered=False)
    log.info(f"Inserted {len(records)} chunks ✓")
    return len(records)


# ════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH
# ════════════════════════════════════════════════════════════════════════════

def semantic_search(
    query:           str,
    limit:           int   = SEARCH_LIMIT,
    score_threshold: float = SCORE_THRESHOLD,
    metadata_filter: Optional[Dict] = None,
    user_id:         Optional[str]  = None,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []

    col          = get_collection()
    query_vector = embed_one(query)
    num_cands    = max(limit * 10, 50)

    pipeline: List[Dict] = [
        {"$vectorSearch": {
            "index":         INDEX_NAME,
            "path":          "embedding",
            "queryVector":   query_vector,
            "numCandidates": num_cands,
            "limit":         limit * 2,
        }},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$match":     {"score": {"$gte": score_threshold}}},
    ]
    if user_id:
        pipeline.append({"$match": {"user_id": user_id}})
    if metadata_filter:
        pipeline.append({"$match": metadata_filter})
    pipeline += [
        {"$project": {"text": 1, "score": 1, "metadata": 1, "_id": 0}},
        {"$sort":    {"score": -1}},
        {"$limit":   limit},
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
    col     = get_collection()
    query   = {"user_id": user_id} if user_id else {}
    total   = col.count_documents(query)
    sources = col.distinct("metadata.source", query)
    return {
        "total_chunks":   total,
        "unique_sources": len(sources),
        "sources":        sources[:20],
        "database":       DATABASE_NAME,
        "collection":     COLLECTION_NAME,
        "index":          INDEX_NAME,
        "embed_model":    LOCAL_EMBED_MODEL,
        "embed_dims":     get_embedding_dims() if _dims else 384,
    }