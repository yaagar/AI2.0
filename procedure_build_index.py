# scripts/procedure_build_index.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pypdf import PdfReader  # pypdf extraction API :contentReference[oaicite:3]{index=3}

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

RAW_DIR = Path("data_procedure_rag/raw")
CHUNKS_DIR = Path("data_procedure_rag/chunks")
INDEX_DIR = Path("data_procedure_rag/index")


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    text: str


def pdf_to_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return pages


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> Iterable[str]:
    t = " ".join(text.split())
    if not t:
        return
    i = 0
    while i < len(t):
        yield t[i : i + chunk_size]
        i += max(1, chunk_size - overlap)


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        pages = pdf_to_pages(pdf)
        for page_idx, page_text in enumerate(pages, start=1):
            for j, ct in enumerate(chunk_text(page_text)):
                cid = f"{pdf.name}::p{page_idx}::c{j}"
                chunks.append(
                    Chunk(
                        id=cid,
                        source=pdf.name,
                        page=page_idx,
                        text=ct,
                    )
                )
    return chunks


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks()
    out_jsonl = CHUNKS_DIR / "chunks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    texts = [c.text for c in chunks]
    metas = [{"id": c.id, "source": c.source, "page": c.page} for c in chunks]

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    vecs = embedder.embed_documents(texts)
    V = np.array(vecs, dtype=np.float32)

    # normalize for cosine similarity
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V = V / norms

    np.savez_compressed(INDEX_DIR / "index.npz", vectors=V, metas=np.array(metas, dtype=object))
    print(f"Saved {len(chunks)} chunks to {out_jsonl}")
    print(f"Saved vectors to {INDEX_DIR / 'index.npz'}")


if __name__ == "__main__":
    main()