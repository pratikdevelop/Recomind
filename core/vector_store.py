"""
core/vector_store.py — MongoDB Atlas Vector Store
Embedding backends:
  • Docker Model Runner  → ai/nomic-embed-text-v1.5:latest  (your Docker model)
  • sentence-transformers → BAAI/bge-small-en-v1.5           (local fallback)
"""

import os
import time
import hashlib
import logging
import requests
from typing import List, Dict, Any, Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.operations import SearchIndexModel

# ── Silence noisy third-party loggers ────────────────────────────────────────
for _lib in ("httpx", "httpcore", "urllib3", "sentence_transformers",
             "transformers", "huggingface_hub", "filelock", "hpack"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MONGODB_URI     = os.getenv("MONGODB_URI",     "")
DATABASE_NAME   = os.getenv("DB_NAME",         "knowledge_base")
COLLECTION_NAME = os.getenv("COLLECTION",      "docs")
INDEX_NAME      = os.getenv("INDEX_NAME",      "vector_idx")

# Embedding backend: 'docker' uses nomic-embed via Docker Model Runner
#                   'local'  uses sentence-transformers (no Docker needed)
EMBED_BACKEND   = os.getenv("EMBED_BACKEND",   "docker")

# Docker Model Runner embedding settings
DOCKER_URL         = os.getenv("DOCKER_URL",         "http://localhost:12434")
DOCKER_EMBED_MODEL = os.getenv("DOCKER_EMBED_MODEL", "nomic-embed-text-v1.5")

# Local sentence-transformers fallback
LOCAL_EMBED_MODEL  = os.getenv("EMBED_MODEL",        "BAAI/bge-small-en-v1.5")

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",    "450"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "80"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.68"))
SEARCH_LIMIT    = int(os.getenv("SEARCH_LIMIT",   "5"))

# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING  — Docker Model Runner (nomic-embed-text)
# ════════════════════════════════════════════════════════════════════════════

_docker_dims: Optional[int] = None   # cached after first call

def _docker_embed(texts: List[str]) -> List[List[float]]:
    """
    Call Docker Model Runner's OpenAI-compatible /v1/embeddings endpoint.
    Model: ai/nomic-embed-text-v1.5:latest  (768-dim, F16, 260 MB)
    """
    global _docker_dims
    url = f"{DOCKER_URL}/engines/llama.cpp/v1/embeddings"
    try:
        resp = requests.post(url, json={"model": DOCKER_EMBED_MODEL, "input": texts}, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        if _docker_dims is None and vecs:
            _docker_dims = len(vecs[0])
            log.info(f"Docker embed model ready — dim={_docker_dims}")
        return vecs
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Docker Model Runner not reachable at port 12434. "
            "Make sure Docker Desktop is running and Model Runner is enabled, "
            "or set EMBED_BACKEND=local in .env to use sentence-transformers instead."
        )
    except Exception as exc:
        raise RuntimeError(f"Docker embedding error: {exc}")


def _get_docker_dims() -> int:
    global _docker_dims
    if _docker_dims is None:
        _docker_embed(["warmup"])   # one call to discover dimension
    return _docker_dims


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING  — Local sentence-transformers (fallback)
# ════════════════════════════════════════════════════════════════════════════

_local_embedder = None
_local_dims: Optional[int] = None

def _get_local_embedder():
    global _local_embedder, _local_dims
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading local embed model: {LOCAL_EMBED_MODEL} …")
        _local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
        _local_dims     = _local_embedder.get_sentence_embedding_dimension()
        log.info(f"Local embed model ready — dim={_local_dims}")
    return _local_embedder


def _local_embed(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    emb = _get_local_embedder()
    out = []
    for i in range(0, len(texts), batch_size):
        vecs = emb.encode(texts[i:i+batch_size], normalize_embeddings=True,
                          show_progress_bar=False)
        out.extend(vecs.tolist())
    return out


def _get_local_dims() -> int:
    _get_local_embedder()
    return _local_dims


# ════════════════════════════════════════════════════════════════════════════
# UNIFIED EMBEDDING API
# ════════════════════════════════════════════════════════════════════════════

def embed_batch(texts: List[str]) -> List[List[float]]:
    if EMBED_BACKEND == "docker":
        return _docker_embed(texts)
    return _local_embed(texts)


def embed_one(text: str) -> List[float]:
    return embed_batch([text])[0]


def get_embedding_dims() -> int:
    if EMBED_BACKEND == "docker":
        return _get_docker_dims()
    return _get_local_dims()


def warm_up_embedder() -> None:
    """Call on startup to initialise the embedder and log which backend is used."""
    backend_label = (
        f"Docker ({DOCKER_EMBED_MODEL})" if EMBED_BACKEND == "docker"
        else f"Local ({LOCAL_EMBED_MODEL})"
    )
    log.info(f"Embedding backend: {backend_label}")
    dims = get_embedding_dims()
    log.info(f"Embedder ready — dim={dims}")


# ════════════════════════════════════════════════════════════════════════════
# MONGODB CONNECTION
# ════════════════════════════════════════════════════════════════════════════

_client     = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI not set in .env")
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
# VECTOR INDEX MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def ensure_index(timeout: int = 120) -> None:
    col  = get_collection()
    dims = get_embedding_dims()

    existing = list(col.list_search_indexes())
    if any(idx["name"] == INDEX_NAME for idx in existing):
        # Verify the stored dimension matches current embedder
        for idx in existing:
            if idx["name"] == INDEX_NAME:
                stored_dims = (
                    idx.get("latestDefinition", {})
                       .get("fields", [{}])[0]
                       .get("numDimensions")
                )
                if stored_dims and stored_dims != dims:
                    log.warning(
                        f"⚠️  Index has {stored_dims} dims but current embedder uses {dims} dims. "
                        f"Drop the '{INDEX_NAME}' index in Atlas and restart to rebuild it."
                    )
        log.info(f"Index '{INDEX_NAME}' exists ✓")
        return

    log.info(f"Creating vector index '{INDEX_NAME}' (dims={dims}) …")
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
    raise TimeoutError(f"Index not queryable after {timeout}s — check Atlas dashboard.")


# ════════════════════════════════════════════════════════════════════════════
# DOCUMENT INGESTION
# ════════════════════════════════════════════════════════════════════════════

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def insert_documents(documents: List[Dict[str, Any]],
                     skip_duplicates: bool = True,
                     user_id: Optional[str] = None) -> int:
    """
    user_id: if provided, every chunk is tagged with this user_id.
             Pass str(current_user.id) from the API route.
    """
    col          = get_collection()
    chunk_texts  = []
    chunk_metas  = []
    chunk_hashes = []

    for doc in documents:
        content = (doc.get("content") or "").strip()
        if not content:
            continue
        h = _sha256(content)
        # Scope duplicate check to the same user
        query = {"content_hash": h}
        if user_id:
            query["user_id"] = user_id
        if skip_duplicates and col.count_documents(query, limit=1):
            log.info(f"  ↳ Duplicate skipped: {doc.get('metadata',{}).get('source','?')}")
            continue
        for chunk in chunk_text(content):
            chunk_texts.append(chunk)
            meta = dict(doc.get("metadata", {}))
            if user_id:
                meta["user_id"] = user_id
            chunk_metas.append(meta)
            chunk_hashes.append(h)

    if not chunk_texts:
        return 0

    log.info(f"Embedding {len(chunk_texts)} chunks via {EMBED_BACKEND} …")
    embeddings = embed_batch(chunk_texts)

    records = [
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
    # Scope to this user's documents only
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
        "total_chunks":    total,
        "unique_sources":  len(sources),
        "sources":         sources[:20],
        "database":        DATABASE_NAME,
        "collection":      COLLECTION_NAME,
        "index":           INDEX_NAME,
        "embed_backend":   EMBED_BACKEND,
        "embed_model":     DOCKER_EMBED_MODEL if EMBED_BACKEND == "docker" else LOCAL_EMBED_MODEL,
        "embed_dims":      get_embedding_dims(),
    }