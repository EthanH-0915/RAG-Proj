"""Hybrid retrieval: exact ID lookup + full-text + vector search, fused with RRF.

Each arm ranks its own candidates, and reciprocal rank fusion scores a chunk as
sum(1 / (RRF_K + rank)) over the arms that found it, so cosine distances and
ts_rank_cd scores never have to be put on one scale.

    id      ATT&CK / CVE IDs pulled out of the question, matched exactly on
            source_id. Sub-techniques of a parent ID are left to fts, which
            finds them through their "Sub-technique of: T1059" line; adding them
            here too gave each child three votes and buried the parent itself.
    fts     the question's lexemes OR-ed together over `tsv` (built from the
            full content), ranked by ts_rank_cd. AND-ing them, as
            websearch_to_tsquery does, leaves natural-language questions with
            almost no matches.
    vector  cosine distance over the embeddings, as in query.py.

CVE IDs only work through the id arm: the english parser splits
'CVE-2021-44228' into 'cve', '-2021' and '-44228'.

With pin_ids=True, exact ID matches rank first whatever their fused score.
Without it, a chunk ranked first by both fts and vector (2/61) outranks an exact
ID match that only the id arm found (1/61).
"""

from __future__ import annotations

import re
from typing import Sequence

from . import config, db, embed

RRF_K = 60
DEPTH = 50  # candidates each arm contributes before fusion
ARMS = ("id", "fts", "vector")

# Only the ID kinds that are (or will be) chunk source_ids: techniques, groups,
# mitigations and CVEs. A tactic or software ID would never match a chunk.
ID_PATTERN = re.compile(r"\b(T\d{4}(?:\.\d{3})?|[GM]\d{4}|CVE-\d{4}-\d{4,7})\b", re.IGNORECASE)

FILTERS = """
           AND (%(doc_type)s::text IS NULL OR doc_type = %(doc_type)s)
           AND (%(platform)s::text IS NULL OR metadata->'platforms' ? %(platform)s)"""

ARM_SQL = {
    "id": f"""
  SELECT id, 'id' AS arm, row_number() OVER (ORDER BY source_id) AS rank, %(pin)s AS pinned
    FROM chunks
   WHERE embedding_model = %(model)s AND source_id = ANY(%(ids)s::text[]){FILTERS}""",
    "fts": f"""
  SELECT id, 'fts' AS arm, row_number() OVER (ORDER BY r DESC) AS rank, false AS pinned
    FROM (SELECT id, ts_rank_cd(tsv, tq) AS r
            FROM chunks,
                 (SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')::tsquery AS tq) q
           WHERE embedding_model = %(model)s AND tsv @@ tq{FILTERS}
           ORDER BY r DESC
           LIMIT %(depth)s) s""",
    "vector": f"""
  SELECT id, 'vector' AS arm, row_number() OVER (ORDER BY d) AS rank, false AS pinned
    FROM (SELECT id, embedding <=> %(q)s AS d
            FROM chunks
           WHERE embedding_model = %(model)s{FILTERS}
           ORDER BY d
           LIMIT %(depth)s) s""",
}

SEARCH = """
WITH ranked AS ({ranked}
),
fused AS (
  SELECT id,
         sum(1.0 / (%(rrf_k)s + rank))::float AS score,
         bool_or(pinned) AS pinned,
         array_agg(arm || ':' || rank ORDER BY arm) AS arms
    FROM ranked
   GROUP BY id
)
SELECT c.source_id, c.doc_type, c.metadata->>'name' AS name,
       f.score AS similarity, f.pinned, f.arms, c.content
  FROM fused f JOIN chunks c USING (id)
 ORDER BY f.pinned DESC, f.score DESC, c.source_id
 LIMIT %(k)s
"""


def extract_ids(question: str) -> list[str]:
    """IDs in the order they appear, upper-cased and de-duplicated."""
    return list(dict.fromkeys(m.upper() for m in ID_PATTERN.findall(question)))


def search(
    question: str,
    k: int = 5,
    doc_type: str | None = None,
    platform: str | None = None,
    *,
    arms: Sequence[str] = ARMS,
    pin_ids: bool = False,
) -> list[dict]:
    """Ranked hits like query.search, with `similarity` holding the RRF score
    and `arms` listing which arms found each chunk, at what rank."""
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {ARMS}")

    ids = extract_ids(question)
    active = [a for a in arms if a != "id" or ids]
    if not active:
        return []

    params = {
        "model": config.EMBEDDING_MODEL,
        "doc_type": doc_type,
        "platform": platform,
        "ids": ids,
        "text": question,
        "pin": pin_ids,
        "depth": DEPTH,
        "rrf_k": RRF_K,
        "k": k,
    }
    if "vector" in active:
        params["q"] = embed.encode_one(question)
    sql = SEARCH.format(ranked="\n UNION ALL".join(f"\n({ARM_SQL[a]})" for a in active))

    with db.connect() as conn, conn.cursor() as cur:
        db.ensure_model(cur)
        cur.execute(sql, params)
        columns = [c.name for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
