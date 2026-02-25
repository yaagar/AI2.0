# tools/kanoon_search_tool.py
# POST-only Indian Kanoon search tool (no RAG, no embeddings). Clean, minimal, reliable.

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Config
# -----------------------------

BASE_URL = os.getenv("IK_BASE_URL", "https://api.indiankanoon.org").rstrip("/")
IK_TOKEN = (os.getenv("IK_TOKEN") or "").strip()
if not IK_TOKEN:
    raise RuntimeError("Missing IK_TOKEN in .env")

TIMEOUT_S = int(os.getenv("IK_TIMEOUT_S", "30"))

HEADERS = {
    "Authorization": f"Token {IK_TOKEN}",
    "Accept": "application/json",
    # DO NOT set Content-Type; we send form-encoded by default via `data=...`
}

# Endpoints (these paths are what your working code implies)
SEARCH_PATH = "/search/"         # requires trailing slash in your working setup
DOC_PATH = "/doc/{docid}/"
DOCMETA_PATH = "/docmeta/{docid}/"


# -----------------------------
# Types
# -----------------------------

@dataclass(frozen=True)
class SearchResult:
    tid: int
    title: str
    docsource: Optional[str] = None
    publishdate: Optional[str] = None
    snippet: Optional[str] = None


@dataclass(frozen=True)
class SearchResponse:
    query: str
    pagenum: int
    found: Optional[int]
    returned: int
    results: List[SearchResult]


# -----------------------------
# Internal HTTP helper
# -----------------------------

def _post_form(path: str, form: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, headers=HEADERS, data=form, timeout=TIMEOUT_S)
    if r.status_code >= 400:
        raise requests.HTTPError(
            f"{r.status_code} {r.reason} for {url} | body={r.text[:700]}",
            response=r,
        )
    return r.json()


# -----------------------------
# Public API: search + fetch
# -----------------------------

def kanoon_search(query: str, pagenum: int = 0) -> SearchResponse:
    """
    POST /search/ with form fields:
      formInput: query
      pagenum: page number as string
    """
    q = (query or "").strip()
    if not q:
        return SearchResponse(query="", pagenum=pagenum, found=0, returned=0, results=[])

    data = _post_form(SEARCH_PATH, {"formInput": q, "pagenum": str(int(pagenum))})

    docs = data.get("docs") or []
    found = data.get("found")

    results: List[SearchResult] = []
    for d in docs:
        tid = d.get("tid")
        if tid is None:
            continue
        try:
            tid_i = int(tid)
        except Exception:
            continue

        results.append(
            SearchResult(
                tid=tid_i,
                title=str(d.get("title") or ""),
                docsource=d.get("docsource"),
                publishdate=d.get("publishdate"),
                snippet=d.get("snippet") or d.get("headline"),
            )
        )

    return SearchResponse(
        query=q,
        pagenum=int(pagenum),
        found=int(found) if isinstance(found, (int, float, str)) and str(found).isdigit() else found,
        returned=len(results),
        results=results,
    )


def kanoon_doc(docid: int) -> Dict[str, Any]:
    """
    POST /doc/<docid>/
    """
    return _post_form(DOC_PATH.format(docid=int(docid)), {})


def kanoon_docmeta(docid: int) -> Dict[str, Any]:
    """
    POST /docmeta/<docid>/
    """
    return _post_form(DOCMETA_PATH.format(docid=int(docid)), {})


# -----------------------------
# Tool-style wrapper (what your agent/tool layer should call)
# -----------------------------

def search_tool(query: str, pagenum: int = 0, top_n: int = 10) -> Dict[str, Any]:
    """
    Returns JSON-serializable dict:
      {
        query, pagenum, found, returned,
        results: [{tid,title,docsource,publishdate,snippet}, ...]
      }
    """
    resp = kanoon_search(query=query, pagenum=pagenum)
    limited = resp.results[: max(0, int(top_n))]

    out = asdict(resp)
    out["results"] = [asdict(r) for r in limited]
    out["returned"] = len(limited)
    return out


# -----------------------------
# Optional: quick manual test
# -----------------------------
if __name__ == "__main__":
    q = input("Query: ").strip()
    print(search_tool(q, pagenum=0, top_n=10))