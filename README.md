# Supertank

Personal knowledge reservoir for algo trading research. Ingest papers, articles, and posts into a searchable index with automatic keyword tagging and semantic embeddings. Used by the Algo Trading Agent Army — August queries it for research context.

## Architecture

```
Source → Extract → Dedup (SHA-256) → KeyBERT keywords → Summary → SQLite index → ChromaDB embed
```

**Storage layers:**
- **SQLite** (`index.db`) — structured metadata: title, source, keywords, summary, content hash
- **ChromaDB** (`supertank` collection via RAG service `:8001`) — semantic vector search over chunked documents
- **Raw files** (`raw/`) — original extracted text, content-addressed

## Quick Start

```bash
# Ingest a PDF
python scripts/add.py ~/papers/paper.pdf

# Ingest a URL
python scripts/add.py https://arxiv.org/abs/2602.11708

# Ingest from stdin
echo "some text" | python scripts/add.py --stdin --title "My Note"

# With manual keywords
python scripts/add.py paper.pdf --keywords "momentum,crypto,systematic"

# Keyword search
python scripts/add.py --search "cryptocurrency trading"

# Semantic search via RAG service
curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"collection":"supertank","query":"market momentum","n_results":5}'
```

## Scripts

| File | Purpose |
|------|---------|
| `scripts/add.py` | Full ingestion pipeline (~320 lines): extract → dedup → tag → index → embed |
| `scripts/db.py` | SQLite schema and CRUD: `insert_document()`, `search()`, `get_db()` |

## Dependencies

All installed in `~/llm-server/venv` (Python 3.14):

- **keybert** — automatic keyword extraction (uses `all-MiniLM-L6-v2`)
- **pymupdf** (fitz) — PDF text extraction
- **beautifulsoup4** + **trafilatura** — HTML/article extraction from URLs
- **sentence-transformers** — embedding model backbone for KeyBERT
- **httpx** — HTTP client for RAG service calls
- **chromadb** — vector storage backend (accessed via RAG service)

**External services:**
- **RAG service** (`localhost:8001`) — Flask app at `~/llm-server/rag_service.py`, supports `collection` param on `/add` and `/search` endpoints

## Ingestion Pipeline Details

1. **Extract** — PyMuPDF for PDFs, curl+BS4/trafilatura for URLs, direct read for text files
2. **Dedup** — SHA-256 content hash; duplicates merge keywords and source URLs
3. **Keyword tagging** — KeyBERT extracts top-10 keyphrases (1-2 gram, first 5000 chars)
4. **Summary** — Extractive: first ~300 chars cut at sentence boundary
5. **Raw storage** — Text saved to `raw/{hash}_{title}.txt`
6. **SQLite** — Metadata row inserted with all fields
7. **ChromaDB** — Text chunked (1000 chars, 200 overlap), embedded into `supertank` collection

## Telegram Integration

The `supertank` Hermes skill provides `/st` commands:

```
/st <url|file>                    → ingest with auto-keywords
/st kw:kw1,kw2 <url|file>        → ingest with manual + auto keywords
/st search <query>                → keyword search (SQLite)
/st find <query>                  → semantic search (ChromaDB)
/st <url> title:"Custom Title"    → ingest with custom title
```

Skill location: `~/.hermes/skills/data-science/supertank/SKILL.md`

## Known Limitations

- SSRN blocks automated fetch (Cloudflare) — download PDFs manually
- OCR not wired yet (tesseract installed)
- Figure/chart extraction: planned, not implemented

## Related Projects

- **Algo Trading Agent Army** (`~/algo-trading-army/`) — 5-agent pipeline; August (strategist) queries Supertank
- **RAG Service** (`~/llm-server/rag_service.py`) — shared ChromaDB backend with multi-collection support
