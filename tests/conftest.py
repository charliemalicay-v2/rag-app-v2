"""Shared fixtures for ingestion tests."""

import os

import psycopg2
import pytest
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.sql import Identifier, SQL

from rag_app.db import db_params

load_dotenv()


@pytest.fixture(scope="session")
def db_connection():
    conn = psycopg2.connect(**db_params())
    register_vector(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def table_name():
    return os.getenv("TABLE_NAME", "documents_v2")


@pytest.fixture(scope="session")
def inserted_ids():
    return []


@pytest.fixture(scope="session")
def cleanup(db_connection, inserted_ids, table_name):
    yield
    if inserted_ids:
        with db_connection.cursor() as cur:
            cur.execute(
                SQL("DELETE FROM {} WHERE id = ANY(%s)").format(
                    Identifier(table_name)
                ),
                (inserted_ids,),
            )
        db_connection.commit()
