"""Embedding model loading and encoding.

Vectors are L2-normalised, so cosine distance (`<=>`) and inner product agree
and the similarity printed by the query script is just `1 - distance`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from . import config


@lru_cache(maxsize=1)
def get_model(name: str = config.EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = getter()
    if dim != config.EMBEDDING_DIM:
        raise ValueError(
            f"{name} produces {dim}-dimensional vectors but the schema column is "
            f"vector({config.EMBEDDING_DIM}); update schema.sql and re-create the table."
        )
    return model


def encode(texts: Sequence[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
    return get_model().encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=progress,
        normalize_embeddings=True,
    )


def encode_one(text: str) -> np.ndarray:
    return encode([text])[0]
