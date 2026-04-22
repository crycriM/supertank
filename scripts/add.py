"""Supertank add — ingest documents into the knowledge reservoir.

Usage:
    python add.py <url|filepath> [--keywords kw1,kw2] [--title "Custom Title"]
    echo "some text" | python add.py --stdin --title "My Note"

Pipeline: extract → hash dedup → KeyBERT tag → summarize → SQLite index → ChromaDB embed
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from db import insert_document, search

VENV_PYTHON = Path.home() / "llm-server" / "venv" / "bin" / "python3"
RAW_DIR = Path.home() / "supertank" / "raw"

# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> tuple[str, int]:
    """Extract text from a PDF using PyMuPDF. Returns (text, page_count)."""
    import fitz
    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    text = "\n\n".join(pages)
    page_count = len(doc)
    doc.close()
    return text, page_count


def extract_text_from_url(url: str) -> tuple[str, str]:
    """Fetch and extract readable text from a URL. Returns (text, title)."""
    import subprocess
    from bs4 import BeautifulSoup

    result = subprocess.run(
        ["curl", "-sL", "--max-time", "30",
         "-H", "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
         url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    html = result.stdout
    soup = BeautifulSoup(html, "html.parser")

    # Try trafilatura for better extraction if available
    try:
        from trafilatura import extract as traf_extract
        text = traf_extract(html) or ""
        if text and len(text) > 200:
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            return text, title
    except ImportError:
        pass

    # Fallback: brute-force BS4 extraction
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = soup.get_text(separator="\n", strip=True)
    return text, title


def extract_text_from_file(filepath: str) -> tuple[str, int]:
    """Extract text from a file. Returns (text, char_count)."""
    p = Path(filepath)
    if p.suffix.lower() == ".pdf":
        text, page_count = extract_text_from_pdf(str(p))
        return text, page_count
    else:
        text = p.read_text(encoding="utf-8", errors="replace")
        return text, 0


# ---------------------------------------------------------------------------
# KeyBERT keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(text: str, n: int = 10) -> list[str]:
    """Extract keywords using KeyBERT with all-MiniLM-L6-v2."""
    from keybert import KeyBERT
    kw_model = KeyBERT()  # defaults to all-MiniLM-L6-v2
    # Use first 5000 chars to keep it fast
    chunk = text[:5000]
    keywords = kw_model.extract_keywords(
        chunk,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=n,
    )
    return [kw for kw, score in keywords]


# ---------------------------------------------------------------------------
# Simple extractive summary (no LLM needed)
# ---------------------------------------------------------------------------

def generate_summary(text: str, max_chars: int = 300) -> str:
    """Generate a rough summary by taking first meaningful paragraph(s)."""
    # For now: first ~300 chars of clean text, cut at sentence boundary
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    # Try to end at last sentence
    for sep in ".!?":
        last = cut.rfind(sep)
        if last > max_chars // 2:
            cut = cut[: last + 1]
            break
    return cut


# ---------------------------------------------------------------------------
# Content hashing for dedup
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ChromaDB embedding
# ---------------------------------------------------------------------------

def embed_to_chroma(doc_id: int, text: str, metadata: dict):
    """Embed document chunks into ChromaDB via local RAG service (:8001)."""
    import httpx

    # Chunk text into ~1000 char segments with overlap
    chunk_size = 1000
    overlap = 200
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if len(chunk.strip()) > 50:  # skip tiny fragments
            chunks.append(chunk)

    if not chunks:
        return 0

    ids = [f"supertank-{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

    # Use the RAG service's /add endpoint in batches of 100
    for batch_start in range(0, len(ids), 100):
        batch_end = batch_start + 100
        resp = httpx.post("http://localhost:8001/add", json={
            "collection": "supertank",
            "documents": chunks[batch_start:batch_end],
            "ids": ids[batch_start:batch_end],
            "metadatas": metadatas[batch_start:batch_end],
        }, timeout=60)
        resp.raise_for_status()

    return len(chunks)


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(
    source: str,
    title: str = "",
    extra_keywords: list[str] | None = None,
    is_stdin: bool = False,
) -> dict:
    """Ingest a document into Supertank. Returns result dict."""

    # 1. Extract text
    if is_stdin:
        text = sys.stdin.read()
        doc_type = "text"
        source_url = ""
        page_count = 0
        if not title:
            title = f"Note {Path(source).stem}" if source else "Untitled note"
    elif source.startswith("http"):
        text, extracted_title = extract_text_from_url(source)
        doc_type = "url"
        source_url = source
        page_count = 0
        if not title:
            title = extracted_title
    else:
        filepath = Path(source)
        if not filepath.exists():
            return {"error": f"File not found: {source}"}
        text, page_count = extract_text_from_file(str(filepath))
        doc_type = filepath.suffix.lstrip(".").lower()
        if doc_type == "pdf":
            doc_type = "pdf"
        source_url = ""
        if not title:
            title = filepath.stem.replace("-", " ").replace("_", " ").title()

    if not text.strip():
        return {"error": "No text extracted from source"}

    char_count = len(text)

    # 2. Hash for dedup
    chash = content_hash(text)

    # 3. Keywords (KeyBERT)
    try:
        auto_keywords = extract_keywords(text, n=10)
    except Exception as e:
        print(f"  [warn] KeyBERT failed: {e}", file=sys.stderr)
        auto_keywords = []

    if extra_keywords:
        # Merge, user keywords first
        all_keywords = list(extra_keywords) + [k for k in auto_keywords if k not in extra_keywords]
    else:
        all_keywords = auto_keywords

    # 4. Summary
    summary = generate_summary(text)

    # 5. Save raw content
    safe_title = title[:80].replace("/", "-").replace(" ", "_")
    raw_path = RAW_DIR / f"{chash[:12]}_{safe_title}.txt"
    raw_path.write_text(text, encoding="utf-8")

    # 6. SQLite index
    row_id = insert_document(
        filepath=str(raw_path),
        title=title[:500],
        source_url=source_url,
        doc_type=doc_type,
        keywords=",".join(all_keywords),
        summary=summary,
        content_hash=chash,
        page_count=page_count,
        char_count=char_count,
    )

    if row_id is None:
        return {"status": "dedup", "hash": chash[:12], "title": title}

    # 7. ChromaDB embedding
    try:
        n_chunks = embed_to_chroma(
            doc_id=row_id,
            text=text,
            metadata={
                "title": title[:200],
                "doc_type": doc_type,
                "source_url": source_url,
                "keywords": ",".join(all_keywords),
            },
        )
    except Exception as e:
        print(f"  [warn] ChromaDB embed failed: {e}", file=sys.stderr)
        n_chunks = 0

    return {
        "status": "ok",
        "id": row_id,
        "title": title,
        "type": doc_type,
        "keywords": all_keywords,
        "chars": char_count,
        "pages": page_count,
        "chunks": n_chunks,
    }


def main():
    parser = argparse.ArgumentParser(description="Supertank: add document to knowledge reservoir")
    parser.add_argument("source", nargs="?", help="URL or filepath to ingest")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--keywords", default="", help="Comma-separated extra keywords")
    parser.add_argument("--title", default="", help="Override document title")
    parser.add_argument("--search", default="", help="Search existing index instead")
    args = parser.parse_args()

    if args.search:
        results = search(query=args.search)
        for r in results:
            print(f"  [{r['id']}] {r['title']}")
            print(f"      type={r['doc_type']}  keywords={r['keywords']}")
            print(f"      {r['summary'][:120]}...")
            print()
        return

    if not args.source and not args.stdin:
        parser.error("Provide a source (URL/filepath) or --stdin")

    source = args.source or "stdin"
    extra_kw = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None

    print(f"Supertank: ingesting {source}...")
    result = ingest(source, title=args.title, extra_keywords=extra_kw, is_stdin=args.stdin)

    if result.get("error"):
        print(f"  ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    elif result["status"] == "dedup":
        print(f"  Duplicate (already indexed): {result['title']}")
    else:
        print(f"  ✓ {result['title']}")
        print(f"    type={result['type']}  chars={result['chars']}  pages={result['pages']}")
        print(f"    keywords: {', '.join(result['keywords'])}")
        print(f"    chunks embedded: {result['chunks']}")


if __name__ == "__main__":
    main()
