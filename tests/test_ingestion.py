"""Tests for vector DB ingestion pipeline.

Required environment variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, TABLE_NAME
"""

from pathlib import Path

import numpy as np
import pytest
from psycopg2.sql import Identifier, SQL
from pypdf import PdfReader

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "1"


# ── Helpers ─────────────────────────────────────────────────────

def read_pdf(filepath):
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() for page in reader.pages)


def read_txt(filepath):
    return Path(filepath).read_text(encoding="utf-8")


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── File reading ────────────────────────────────────────────────

class TestFileReading:
    def test_read_pdf(self):
        pdf_files = list(DATASET_DIR.glob("*.pdf"))
        assert len(pdf_files) > 0, "No PDF files found in datasets/1/"
        content = read_pdf(pdf_files[0])
        assert isinstance(content, str)
        assert len(content.strip()) > 0

    def test_read_txt(self):
        txt_files = list(DATASET_DIR.glob("*.txt"))
        assert len(txt_files) > 0, "No TXT files found in datasets/1/"
        content = read_txt(txt_files[0])
        assert isinstance(content, str)
        assert len(content.strip()) > 0


# ── Chunking ────────────────────────────────────────────────────

class TestChunking:
    def test_chunk_text_produces_multiple_chunks(self):
        text = "Hello world. " * 100
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_small_input(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text


# ── Embedding ───────────────────────────────────────────────────

class TestEmbedding:
    def test_generate_embedding_shape_and_type(self, embedding_model):
        text = "Test sentence for embedding."
        vec = embedding_model.encode(text)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_similar_texts_have_higher_similarity(self, embedding_model):
        texts = [
            "The dog chased the cat up the tree.",
            "A canine was pursuing a feline.",
            "Quantum mechanics describes subatomic particles.",
        ]
        vecs = embedding_model.encode(texts)
        sim_01 = np.dot(vecs[0], vecs[1]) / (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        )
        sim_02 = np.dot(vecs[0], vecs[2]) / (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[2])
        )
        assert sim_01 > sim_02


# ── Database connectivity & schema ──────────────────────────────

class TestDatabase:
    def test_db_connection(self, db_connection):
        with db_connection.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1

    def test_pgvector_extension_installed(self, db_connection):
        with db_connection.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            assert cur.fetchone() is not None

    def test_documents_v2_table_exists(self, db_connection, table_name):
        with db_connection.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s",
                (table_name,),
            )
            assert cur.fetchone() is not None

    def test_documents_v2_has_required_columns(self, db_connection, table_name):
        required = {"id", "content", "embedding"}
        with db_connection.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s",
                (table_name,),
            )
            existing = {row[0] for row in cur.fetchall()}
        missing = required - existing
        assert not missing, f"Missing required columns: {missing}"


# ── Ingestion pipeline ──────────────────────────────────────────

@pytest.mark.usefixtures("cleanup")
class TestIngestion:
    def test_insert_single_document(
        self, db_connection, embedding_model, inserted_ids, table_name
    ):
        content = "Test document for insertion verification."
        vec = embedding_model.encode(content)
        tbl = Identifier(table_name)

        with db_connection.cursor() as cur:
            cur.execute(
                SQL(
                    "INSERT INTO {} (content, embedding) "
                    "VALUES (%s, %s) RETURNING id"
                ).format(tbl),
                (content, vec),
            )
            row_id = cur.fetchone()[0]
        db_connection.commit()
        inserted_ids.append(row_id)

        assert isinstance(row_id, int)
        assert row_id > 0

    def test_similarity_search_returns_most_relevant(
        self, db_connection, embedding_model, inserted_ids, table_name
    ):
        docs = [
            "Brake pad maintenance and rotor inspection procedures.",
            "Climate control system and AC operation guide.",
            "Tire pressure monitoring and replacement steps.",
        ]
        vecs = embedding_model.encode(docs)
        tbl = Identifier(table_name)
        my_ids = []

        with db_connection.cursor() as cur:
            for doc, vec in zip(docs, vecs):
                cur.execute(
                    SQL(
                        "INSERT INTO {} (content, embedding) "
                        "VALUES (%s, %s) RETURNING id"
                    ).format(tbl),
                    (doc, vec),
                )
                row_id = cur.fetchone()[0]
                my_ids.append(row_id)
                inserted_ids.append(row_id)
        db_connection.commit()

        query = "How do I check and maintain my brake pads?"
        query_vec = embedding_model.encode(query)

        with db_connection.cursor() as cur:
            cur.execute(
                SQL(
                    "SELECT content FROM {} "
                    "WHERE id = ANY(%s) "
                    "ORDER BY embedding <=> %s "
                    "LIMIT 1"
                ).format(tbl),
                (my_ids, query_vec),
            )
            top = cur.fetchone()

        assert top is not None
        assert "brake" in top[0].lower()

    def test_end_to_end_pipeline(
        self, db_connection, embedding_model, inserted_ids, table_name
    ):
        pdf_files = list(DATASET_DIR.glob("*.pdf"))
        assert len(pdf_files) > 0

        content = read_pdf(pdf_files[0])
        chunks = chunk_text(content, chunk_size=500, overlap=50)
        assert len(chunks) > 0

        sample = chunks[:5]
        vecs = embedding_model.encode(sample)
        tbl = Identifier(table_name)
        my_ids = []

        with db_connection.cursor() as cur:
            for chunk, vec in zip(sample, vecs):
                cur.execute(
                    SQL(
                        "INSERT INTO {} (content, embedding) "
                        "VALUES (%s, %s) RETURNING id"
                    ).format(tbl),
                    (chunk, vec),
                )
                row_id = cur.fetchone()[0]
                my_ids.append(row_id)
                inserted_ids.append(row_id)
        db_connection.commit()

        query = sample[0][:100]
        query_vec = embedding_model.encode(query)

        with db_connection.cursor() as cur:
            cur.execute(
                SQL(
                    "SELECT content FROM {} "
                    "WHERE id = ANY(%s) "
                    "ORDER BY embedding <=> %s "
                    "LIMIT 1"
                ).format(tbl),
                (my_ids, query_vec),
            )
            result = cur.fetchone()

        assert result is not None
        assert len(result[0]) > 0
