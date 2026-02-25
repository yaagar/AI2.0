# tools/kanoon_rag_tool.py
# RAG Retriever Tool for Indian Kanoon (POST search/doc/docmeta via kanoon_search_tool.py)
# - Caches docs locally
# - Chunks docs
# - Embeds chunks
# - Stores persistent vector index (JSONL + .npy)
# - Retrieves top chunks by cosine similarity

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup  # pip install beautifulsoup4

from tools.kanoon_search_tool import kanoon_search, kanoon_doc, kanoon_docmeta

load_dotenv()


# -----------------------------
# Storage config
# -----------------------------
DATA_DIR = os.getenv("KANOON_RAG_DIR", "data_kanoon_rag")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")
INDEX_DIR = os.path.join(DATA_DIR, "index")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_META_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")     # metadata + text
INDEX_VEC_PATH = os.path.join(INDEX_DIR, "embeddings.npy")    # float32 matrix (N,d)

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


# -----------------------------
# Tool output contracts
# -----------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    tid: int
    title: str
    chunk_id: str
    score: float
    text: str
    publishdate: Optional[str] = None
    docsource: Optional[str] = None


@dataclass(frozen=True)
class RetrieverOutput:
    query: str
    rewrites_used: List[str]
    tids_considered: List[int]
    returned: int
    results: List[RetrievedChunk]
    notes: List[str]


# -----------------------------
# Helpers: filesystem
# -----------------------------

def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", s)

def _raw_path(tid: int) -> str:
    return os.path.join(RAW_DIR, f"{int(tid)}.json")

def _chunk_path(tid: int) -> str:
    return os.path.join(CHUNK_DIR, f"{int(tid)}.json")

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _file_exists(path: str) -> bool:
    return os.path.exists(path)


# -----------------------------
# Helpers: text cleaning + chunking
# -----------------------------

def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    txt = soup.get_text(separator=" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Query rewrites (no LLM; deterministic)
# -----------------------------

def rewrite_queries(user_query: str, max_q: int = 5) -> List[str]:
    """
    Deterministic rewrite: generates a few variants without calling an LLM.
    Goal: improve recall while staying stable.
    """
    q = (user_query or "").strip()
    if not q:
        return []

    ql = q.lower()
    variants = [q]

    # simple synonyms / legal terms expansion
    swaps = [
        ("termination of pregnancy", "abortion"),
        ("medical termination of pregnancy", "MTP"),
        ("dismissal", "termination"),
        ("agreement", "contract"),
        ("breach", "repudiation"),
    ]
    for a, b in swaps:
        if a in ql:
            variants.append(re.sub(re.escape(a), b, q, flags=re.IGNORECASE))

    # if query contains "section", keep as-is; else add "section" variant
    if "section" not in ql and "act" in ql:
        variants.append(q + " section")

    # dedupe + cap
    out: List[str] = []
    seen = set()
    for v in variants:
        v = re.sub(r"\s+", " ", v).strip()
        if v and v.lower() not in seen:
            out.append(v)
            seen.add(v.lower())
        if len(out) >= max_q:
            break
    return out


# -----------------------------
# Persistent local vector index
# -----------------------------

class LocalVectorIndex:
    """
    Persistent store:
      - chunks.jsonl: {chunk_id, tid, title, publishdate, docsource, text}
      - embeddings.npy: float32 matrix (N,d), normalized rows

    Simple and robust. Later you can swap with FAISS.
    """

    def __init__(self, embed_model: str = EMBED_MODEL):
        self.embedder = OpenAIEmbeddings(model=embed_model)
        self.meta: List[Dict[str, Any]] = []
        self.mat: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        self.meta = []
        if _file_exists(INDEX_META_PATH):
            with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.meta.append(json.loads(line))
        self.mat = None
        if _file_exists(INDEX_VEC_PATH):
            arr = np.load(INDEX_VEC_PATH)
            if arr.size:
                self.mat = arr.astype(np.float32, copy=False)

    def _save_append(self, meta_rows: List[Dict[str, Any]], vec_rows: np.ndarray) -> None:
        # append metadata
        with open(INDEX_META_PATH, "a", encoding="utf-8") as f:
            for r in meta_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # append vectors
        if _file_exists(INDEX_VEC_PATH):
            existing = np.load(INDEX_VEC_PATH).astype(np.float32, copy=False)
            combined = np.vstack([existing, vec_rows]).astype(np.float32, copy=False)
            np.save(INDEX_VEC_PATH, combined)
        else:
            np.save(INDEX_VEC_PATH, vec_rows.astype(np.float32, copy=False))

        # reload
        self._load()

    def has_chunk(self, chunk_id: str) -> bool:
        return any(m["chunk_id"] == chunk_id for m in self.meta)

    def add_chunks(self, chunk_rows: List[Dict[str, Any]]) -> int:
        """
        chunk_rows: list of {chunk_id, tid, title, publishdate, docsource, text}
        Adds only unseen chunk_id.
        """
        new_rows = [r for r in chunk_rows if not self.has_chunk(r["chunk_id"])]
        if not new_rows:
            return 0

        texts = [r["text"] for r in new_rows]
        vecs = self.embedder.embed_documents(texts)
        mat = np.array(vecs, dtype=np.float32)

        # normalize for cosine
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = mat / norms

        self._save_append(new_rows, mat)
        return len(new_rows)

    def search(self, query: str, k: int = 10, restrict_tids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Returns list of (index_in_meta, score) sorted desc.
        """
        if self.mat is None or not self.meta:
            return []

        qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        sims = self.mat @ qv  # (N,)

        if restrict_tids:
            allowed = set(int(t) for t in restrict_tids)
            mask = np.array([int(m["tid"]) in allowed for m in self.meta], dtype=bool)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                return []
            # rank among restricted
            sub = sims[idxs]
            top_local = idxs[np.argsort(-sub)[:k]]
        else:
            top_local = np.argsort(-sims)[:k]

        out = [(int(i), float(sims[int(i)])) for i in top_local]
        return out


# -----------------------------
# Kanoon fetch + cache + chunk + index
# -----------------------------

def fetch_and_cache_tid(tid: int) -> Dict[str, Any]:
    """
    Fetch doc HTML and meta and store raw JSON locally.
    """
    dj = kanoon_doc(int(tid))
    mj = kanoon_docmeta(int(tid))

    raw = {
        "tid": int(tid),
        "doc_html": dj.get("doc", "") or "",
        "title": mj.get("title") or "",
        "publishdate": mj.get("publishdate"),
        "docsource": mj.get("docsource"),
        "meta": mj,
    }
    _write_json(_raw_path(int(tid)), raw)
    return raw

def load_or_fetch_tid(tid: int) -> Dict[str, Any]:
    path = _raw_path(int(tid))
    if _file_exists(path):
        return _read_json(path)
    return fetch_and_cache_tid(int(tid))

def build_chunks_for_tid(raw: Dict[str, Any], max_chunks_per_doc: int = 8) -> List[Dict[str, Any]]:
    """
    Returns list of chunk rows for index:
      {chunk_id, tid, title, publishdate, docsource, text}
    Also caches chunks to CHUNK_DIR.
    """
    tid = int(raw["tid"])
    title = raw.get("title") or ""
    publishdate = raw.get("publishdate")
    docsource = raw.get("docsource")

    text = html_to_text(raw.get("doc_html", ""))
    parts = chunk_text(text, max_chars=1500, overlap=200)[: max_chunks_per_doc]

    rows: List[Dict[str, Any]] = []
    for i, ch in enumerate(parts):
        chunk_id = f"{tid}:{i}:{_sha1(ch[:300])[:10]}"
        rows.append(
            {
                "chunk_id": chunk_id,
                "tid": tid,
                "title": title,
                "publishdate": publishdate,
                "docsource": docsource,
                "text": ch,
            }
        )

    # cache for debugging/inspection
    _write_json(_chunk_path(tid), {"tid": tid, "title": title, "chunks": rows})
    return rows


# -----------------------------
# Main tool: hybrid retrieve
# -----------------------------

def rag_retrieve_tool(
    user_query: str,
    per_query_k: int = 10,
    max_docs: int = 30,
    top_chunks: int = 8,
    max_chunks_per_doc: int = 6,
    min_score: float = 0.20,
    sleep_s: float = 0.10,
) -> Dict[str, Any]:
    """
    Tool entrypoint.
    1) rewrite queries (deterministic)
    2) kanoon_search for each rewrite -> collect tids
    3) fetch/cache doc+meta for tids
    4) chunk + embed + add to index (only new chunks)
    5) embed user query and retrieve top chunks (restricted to tids considered)
    """
    q = (user_query or "").strip()
    if not q:
        out = RetrieverOutput(query="", rewrites_used=[], tids_considered=[], returned=0, results=[], notes=["empty_query"])
        return asdict(out)

    rewrites = rewrite_queries(q, max_q=5)
    notes: List[str] = []
    tids: List[int] = []
    title_by_tid: Dict[int, str] = {}

    # 1) gather tids from Kanoon search
    for rq in rewrites:
        sr = kanoon_search(rq, pagenum=0)  # returns SearchResponse
        for r in sr.results[:per_query_k]:
            if r.tid not in title_by_tid:
                title_by_tid[r.tid] = r.title or ""
                tids.append(int(r.tid))
                if len(tids) >= max_docs:
                    break
        if len(tids) >= max_docs:
            break

    tids = tids[:max_docs]
    notes.append(f"rewrites={len(rewrites)}")
    notes.append(f"tids={len(tids)}")

    if not tids:
        out = RetrieverOutput(query=q, rewrites_used=rewrites, tids_considered=[], returned=0, results=[], notes=notes + ["no_tids_found"])
        return asdict(out)

    # 2) ensure indexed
    index = LocalVectorIndex(embed_model=EMBED_MODEL)

    ingested_chunks = 0
    fetched_docs = 0

    for tid in tids:
        # build chunks, add if new
        raw = load_or_fetch_tid(tid)
        fetched_docs += 1

        chunk_rows = build_chunks_for_tid(raw, max_chunks_per_doc=max_chunks_per_doc)
        ingested_chunks += index.add_chunks(chunk_rows)

        time.sleep(sleep_s)

    notes.append(f"fetched_docs={fetched_docs}")
    notes.append(f"ingested_chunks={ingested_chunks}")

    # 3) retrieve chunks from local index, restricted to tids considered
    hits = index.search(query=q, k=max(top_chunks * 4, 20), restrict_tids=tids)

    # build result list, apply min_score, cap top_chunks
    results: List[RetrievedChunk] = []
    per_tid_cap: Dict[int, int] = {}

    for meta_idx, score in hits:
        if len(results) >= top_chunks:
            break
        if score < min_score:
            continue

        m = index.meta[meta_idx]
        tid = int(m["tid"])

        # diversify: cap 3 chunks per tid
        if per_tid_cap.get(tid, 0) >= 3:
            continue

        results.append(
            RetrievedChunk(
                tid=tid,
                title=str(m.get("title") or ""),
                chunk_id=str(m.get("chunk_id") or ""),
                score=float(score),
                text=str(m.get("text") or ""),
                publishdate=m.get("publishdate"),
                docsource=m.get("docsource"),
            )
        )
        per_tid_cap[tid] = per_tid_cap.get(tid, 0) + 1

    out = RetrieverOutput(
        query=q,
        rewrites_used=rewrites,
        tids_considered=tids,
        returned=len(results),
        results=results,
        notes=notes,
    )
    return {
        "query": out.query,
        "rewrites_used": out.rewrites_used,
        "tids_considered": out.tids_considered,
        "returned": out.returned,
        "results": [asdict(r) for r in out.results],
        "notes": out.notes,
    }


if __name__ == "__main__":
    q = input("Query: ").strip()
    data = rag_retrieve_tool(q)
    print(json.dumps(data, ensure_ascii=False, indent=2))