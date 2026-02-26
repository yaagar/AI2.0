# scripts/procedure_download.py
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

UA = "LawyalAI-ProcedureHarvester/0.1"

SEED_PAGES = [
    "https://ecourts.gov.in/ecourts_home/static/manuals.php",
]

SEED_PDFS = [
    # eCourts eFiling User Manual
    "https://ecourts.gov.in/ecourts_home/static/manuals/efiling-User-manual.pdf",
]

OUT_DIR = Path("data_procedure_rag/raw")


def safe_name(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    fname = url.split("/")[-1] or "download.pdf"
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    return f"{h}__{fname}"


def extract_pdf_links(page_url: str) -> set[str]:
    r = requests.get(page_url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pdfs = set()
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if href.lower().endswith(".pdf"):
            pdfs.add(urljoin(page_url, href))
    return pdfs


def download(url: str) -> Path | None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / safe_name(url)
    if out.exists() and out.stat().st_size > 0:
        return out

    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "pdf" not in ctype and not url.lower().endswith(".pdf"):
        return None

    out.write_bytes(r.content)
    return out


def main():
    all_pdfs = set(SEED_PDFS)
    for page in SEED_PAGES:
        try:
            all_pdfs |= extract_pdf_links(page)
        except Exception as e:
            print(f"[WARN] failed to parse {page}: {e}")

    ok = 0
    for url in sorted(all_pdfs):
        try:
            p = download(url)
            if p:
                ok += 1
                print(f"[OK] {url} -> {p}")
        except Exception as e:
            print(f"[WARN] download failed {url}: {e}")
        time.sleep(0.6)

    print(f"Done. Downloaded {ok} PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()