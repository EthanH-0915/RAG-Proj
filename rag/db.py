"""Database connection with the pgvector type adapters registered."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from . import config


def connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn


def ensure_model(cur: psycopg.Cursor) -> None:
    """Exit with a hint unless the table holds chunks embedded with the configured
    model; a query embedded with another model would return nonsense distances."""
    cur.execute("SELECT embedding_model, count(*) FROM chunks GROUP BY embedding_model")
    present = dict(cur.fetchall())
    if config.EMBEDDING_MODEL not in present:
        raise SystemExit(
            f"No chunks embedded with {config.EMBEDDING_MODEL}.\n"
            f"Models in the table: {present or 'none — run python -m rag.ingest'}"
        )
