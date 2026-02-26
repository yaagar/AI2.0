# tools/procedure_rag_tool.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_DIR = os.getenv("PROCEDURE_RAG_DIR", "data_procedure_rag")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks", "chunks.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "index", "index.npz")

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


@dataclass(frozen=True)
class RetrievedChunk:
    tid: int
    title: str
    chunk_id: str
    score: float
    text: str
    publishdate: Optional[str] = None
    docsource: Optional[str] = None


def _load_chunks_map() -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mp[obj["id"]] = obj
    return mp


def _load_index():
    data = np.load(INDEX_PATH, allow_pickle=True)
    V = data["vectors"].astype(np.float32, copy=False)
    metas = data["metas"]  # array of dict-like objects
    return V, metas


def procedure_retrieve_tool(
    user_query: str,
    top_chunks: int = 8,
    min_score: float = 0.15,
) -> Dict[str, Any]:
    q = (user_query or "").strip()
    if not q:
        return {"query": "", "rewrites_used": [], "tids_considered": [], "returned": 0, "results": [], "notes": ["empty_query"]}

    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(INDEX_PATH):
        return {
            "query": q,
            "rewrites_used": [],
            "tids_considered": [],
            "returned": 0,
            "results": [],
            "notes": [f"missing_files chunks={os.path.exists(CHUNKS_PATH)} index={os.path.exists(INDEX_PATH)}"],
        }

    # Load once per call (fine for now; later you can cache globally)
    V, metas = _load_index()
    chunks = _load_chunks_map()

    embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    qv = np.array(embedder.embed_query(q), dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)

    # cosine similarity (V should already be normalized from build script)
    sims = V @ qv
    idx = np.argsort(-sims)[: max(top_chunks * 4, 20)]

    results: List[RetrievedChunk] = []
    for i in idx:
        score = float(sims[int(i)])
        if score < min_score:
            continue

        meta = metas[int(i)]
        meta = meta.item() if hasattr(meta, "item") else meta  # handle numpy object

        cid = meta.get("id")
        obj = chunks.get(cid)
        if not obj:
            continue

        title = obj.get("source", "")
        page = obj.get("page", None)

        results.append(
            RetrievedChunk(
                tid=0,
                title=f"{title} (page {page})" if page else str(title),
                chunk_id=str(cid),
                score=score,
                text=str(obj.get("text", "")),
                publishdate=None,
                docsource=str(title),
            )
        )
        if len(results) >= top_chunks:
            break

    return {
        "query": q,
        "rewrites_used": [],
        "tids_considered": [],
        "returned": len(results),
        "results": [asdict(r) for r in results],
        "notes": [f"corpus=procedure", f"chunks_file={CHUNKS_PATH}", f"index_file={INDEX_PATH}"],
    }