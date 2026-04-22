"""Supertank SQLite index — structured metadata for all ingested documents."""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path.home() / "supertank" / "index.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    title TEXT,
    source_url TEXT,
    doc_type TEXT NOT NULL DEFAULT 'text',       -- pdf, url, text, image
    keywords TEXT,                                 -- comma-separated tags
    summary TEXT,
    content_hash TEXT,                             -- dedup key (sha256)
    page_count INTEGER,
    char_count INTEGER,
    date_added TEXT NOT NULL,
    date_modified TEXT
);

CREATE INDEX IF NOT EXISTS idx_keywords ON documents(keywords);
CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_date_added ON documents(date_added);
CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash);
"""


def get_db() -> sqlite3.Connection:
    """Get a connection to the supertank index database."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def insert_document(
    filepath: str,
    title: str = "",
    source_url: str = "",
    doc_type: str = "text",
    keywords: str = "",
    summary: str = "",
    content_hash: str = "",
    page_count: int = 0,
    char_count: int = 0,
) -> int | None:
    """Insert a document record. Returns row id or None if duplicate (by content_hash)."""
    db = get_db()
    try:
        # Check dedup
        if content_hash:
            existing = db.execute(
                "SELECT id, filepath FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing:
                # Merge refs: append new filepath/source as additional ref
                existing_sources = db.execute(
                    "SELECT source_url FROM documents WHERE id = ?",
                    (existing["id"],),
                ).fetchone()["source_url"] or ""
                urls = set(filter(None, existing_sources.split(",")))
                if source_url:
                    urls.add(source_url)
                db.execute(
                    "UPDATE documents SET source_url = ?, date_modified = ? WHERE id = ?",
                    (",".join(sorted(urls)), datetime.utcnow().isoformat(), existing["id"]),
                )
                db.commit()
                return None  # dedup, not a new insert

        now = datetime.utcnow().isoformat()
        cursor = db.execute(
            """INSERT OR IGNORE INTO documents
            (filepath, title, source_url, doc_type, keywords, summary,
             content_hash, page_count, char_count, date_added)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (filepath, title, source_url, doc_type, keywords, summary,
             content_hash, page_count, char_count, now),
        )
        db.commit()
        return cursor.lastrowid
    finally:
        db.close()


def search(
    query: str = "",
    doc_type: str = "",
    keywords: str = "",
    since: str = "",
    limit: int = 20,
) -> list[dict]:
    """Search the index by keyword, type, or date."""
    db = get_db()
    try:
        sql = "SELECT * FROM documents WHERE 1=1"
        params = []
        if query:
            sql += " AND (title LIKE ? OR summary LIKE ? OR keywords LIKE ?)"
            q = f"%{query}%"
            params.extend([q, q, q])
        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)
        if keywords:
            for kw in keywords.split(","):
                sql += " AND keywords LIKE ?"
                params.append(f"%{kw.strip()}%")
        if since:
            sql += " AND date_added >= ?"
            params.append(since)
        sql += f" ORDER BY date_added DESC LIMIT {limit}"
        rows = db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


if __name__ == "__main__":
    # Quick sanity check
    db = get_db()
    count = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"Supertank index: {count} documents")
    db.close()
