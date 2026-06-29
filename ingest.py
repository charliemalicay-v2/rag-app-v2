#!/usr/bin/env python3
"""Ingest documents from a dataset folder into the vector DB.

Usage:
    python ingest.py datasets/1/
    python ingest.py datasets/1/ --chunk-size 400 --overlap 80 --batch-size 50 --limit 200
"""

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.sql import Identifier, SQL
from pypdf import PdfReader

load_dotenv()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ingest documents into vector DB")
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Relative path to folder containing .pdf and .txt files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Characters per chunk (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Sliding-window overlap between chunks (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Rows per INSERT batch (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max chunks to process total (for testing, default: no limit)",
    )
    return parser.parse_args(argv)


# ── DB helpers ───────────────────────────────────────────────────────────────

def db_params():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "vector_db_1"),
        "user": os.getenv("DB_USER", "charlie"),
        "password": os.getenv("DB_PASSWORD", "malicay"),
    }


def connect_db():
    conn = psycopg2.connect(**db_params())
    register_vector(conn)
    return conn


def ensure_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
            (table_name,),
        )
        if cur.fetchone() is not None:
            return
        cur.execute(
            SQL(
                "CREATE TABLE {} ("
                "id BIGSERIAL PRIMARY KEY, "
                "content TEXT NOT NULL, "
                "embedding vector(384)"
                ")"
            ).format(Identifier(table_name))
        )
    conn.commit()


# ── File reading ─────────────────────────────────────────────────────────────

FILE_EXTENSIONS = (".pdf", ".txt")


def find_files(dataset_dir):
    path = Path(dataset_dir).resolve()
    if not path.is_dir():
        print(f"Error: '{dataset_dir}' is not a valid directory")
        sys.exit(1)
    files = sorted(
        f for f in path.iterdir() if f.suffix.lower() in FILE_EXTENSIONS
    )
    return files


def read_pdf(filepath):
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() for page in reader.pages)


def read_txt(filepath):
    return Path(filepath).read_text(encoding="utf-8")


def read_file(filepath):
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(filepath)
    elif suffix == ".txt":
        return read_txt(filepath)
    raise ValueError(f"Unsupported file type: {suffix}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── Dedup ─────────────────────────────────────────────────────────────────────

def filter_new_chunks(cur, table_name, chunks, batch_size=500):
    """Return only chunks whose content does not already exist in the table.

    Queries are split into batches to stay within PostgreSQL's parameter
    limit (max 65535 per query).
    """
    if not chunks:
        return chunks
    existing = set()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        placeholders = ",".join("%s" for _ in batch)
        cur.execute(
            SQL("SELECT content FROM {} WHERE content IN ({})").format(
                Identifier(table_name), SQL(placeholders)
            ),
            batch,
        )
        existing.update(row[0] for row in cur.fetchall())
    return [c for c in chunks if c not in existing]


# ── Embedding ─────────────────────────────────────────────────────────────────

def build_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Batch insert ──────────────────────────────────────────────────────────────

def insert_batch(cur, table_name, chunks, embeddings):
    values = [
        (chunk, emb.tolist())
        for chunk, emb in zip(chunks, embeddings)
    ]
    cur.executemany(
        SQL("INSERT INTO {} (content, embedding) VALUES (%s, %s::vector)").format(
            Identifier(table_name)
        ),
        values,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    table_name = os.getenv("TABLE_NAME", "documents_v2")

    files = find_files(args.dataset_dir)
    if not files:
        print(f"No .pdf or .txt files found in '{args.dataset_dir}'")
        sys.exit(1)

    print(f"Found {len(files)} file(s) in '{args.dataset_dir}'")
    print(f"Table: {table_name}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.overlap}")
    print(f"Batch size: {args.batch_size}" + (f", limit: {args.limit}" if args.limit else ""))
    print()

    conn = connect_db()
    ensure_table(conn, table_name)
    embedder = build_embedder()

    total_chunks = 0
    total_new = 0
    total_skipped = 0
    total_files_ok = 0
    total_files_err = 0
    t0 = time.time()

    for idx, filepath in enumerate(files, 1):
        try:
            content = read_file(filepath)
            chunks = chunk_text(content, args.chunk_size, args.overlap)

            if args.limit:
                remaining = args.limit - total_chunks
                if remaining <= 0:
                    break
                chunks = chunks[:remaining]

            with conn.cursor() as cur:
                new_chunks = filter_new_chunks(cur, table_name, chunks)

            n_new = len(new_chunks)
            n_skip = len(chunks) - n_new
            total_chunks += len(chunks)
            total_new += n_new
            total_skipped += n_skip

            if n_new > 0:
                embeddings = embedder.encode(new_chunks, show_progress_bar=False)
                batch_size = args.batch_size
                for i in range(0, n_new, batch_size):
                    batch = new_chunks[i : i + batch_size]
                    batch_embs = embeddings[i : i + batch_size]
                    with conn.cursor() as cur:
                        insert_batch(cur, table_name, batch, batch_embs)
                    conn.commit()

            total_files_ok += 1
            print(
                f"[{idx}/{len(files)}] {filepath.name} → "
                f"{len(chunks)} chunks, {n_new} new"
            )

        except Exception as exc:
            total_files_err += 1
            print(f"[{idx}/{len(files)}] {filepath.name} → ERROR: {exc}")

    elapsed = time.time() - t0
    print()
    print("─" * 50)
    print(f"Finished:  {elapsed:.1f}s")
    print(f"Files:     {total_files_ok} OK, {total_files_err} error(s)")
    print(f"Chunks:    {total_chunks} total, {total_new} new, {total_skipped} duplicate(s)")

    conn.close()


if __name__ == "__main__":
    main()
