"""Hybrid retrieval: exact ID lookup + full-text + vector search, fused with RRF.

Each arm ranks its own candidates, and reciprocal rank fusion scores a chunk as
sum(1 / (RRF_K + rank)) over the arms that found it, so cosine distances and
ts_rank_cd scores never have to be put on one scale.

    id      ATT&CK / CVE IDs pulled out of the question, matched exactly on
            source_id, plus groups the question names by name or alias ("Cozy
            Bear" -> G0016). fts cannot do the latter: ts_rank has no IDF, so
            "threat", "group" and "known", present in every group chunk, count
            as much as "cozy" and "bear". Sub-techniques of a parent ID are left
            to fts, which finds them through their "Sub-technique of: T1059"
            line; adding them here too gave each child three votes and buried
            the parent itself.
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

With expand=True, a question whose wording asks for a related kind of object
follows the relationship IDs in a source chunk's metadata to those objects, and
ranks them right after any pinned ID matches. RELATIONS is the table: "Which
mitigations address password spraying?" goes technique -> `mitigated_by` ->
mitigation chunks. A mitigation's own text rarely names the techniques it
applies to, so none of the three arms finds it from "password spraying" alone.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from . import config, db, embed

RRF_K = 60
DEPTH = 50  # candidates each arm contributes before fusion
ARMS = ("id", "fts", "vector")

# Only the ID kinds that are (or will be) chunk source_ids: techniques, groups,
# mitigations and CVEs. A tactic or software ID would never match a chunk.
ID_PATTERN = re.compile(r"\b(T\d{4}(?:\.\d{3})?|[GM]\d{4}|CVE-\d{4}-\d{4,7})\b", re.IGNORECASE)

# Wording that asks for a related kind of object rather than the one named.
MITIGATION_INTENT = re.compile(
    r"\b(mitigat\w*|prevent\w*|defen[cds]\w*|protect\w*|countermeasures?|reduce the risk)\b",
    re.IGNORECASE,
)
# "Which groups / what threat actors / who uses ...", but not "How do adversaries
# ...", which opens most conceptual questions and asks about the technique.
GROUP_INTENT = re.compile(
    r"\b(which|what)\s+(\w+\s+){0,2}(groups?|actors?|APTs?)\b"
    r"|\bwho\s+(has\s+|have\s+)?(used|uses|use|exploited|exploits|is behind|are behind)\b",
    re.IGNORECASE,
)
# Everything that phrases the group relation, removed before guessing the
# technique: the verbs matter as much as "groups", because every technique chunk
# says "Used by groups:", so a leftover "use" makes fts favour whichever
# technique repeats it most ("Which groups use Kerberoasting?" found Application
# Access Token).
GROUP_WORDING = re.compile(
    r"\b(which|what|who|threat|groups?|actors?|APTs?|ha[sd]|have|us(?:e[sd]?|ing)"
    r"|exploit\w*|abus\w*|is behind|are behind)\b",
    re.IGNORECASE,
)
# "CVE" inside an ID ("CVE-2020-1472") is not wording.
CVE_INTENT = re.compile(r"\b(CVEs?(?!-\d)|vulnerabilit(?:y|ies))\b", re.IGNORECASE)


@dataclass(frozen=True)
class Relation:
    """One link-following rule: a question matching `intent` wants `target`
    chunks, reached through the metadata key listed for each source doc type."""

    target: str
    intent: re.Pattern
    keys: dict[str, str]  # source doc_type -> metadata key holding target IDs
    # Doc type to guess the source from when the question names nothing. None
    # means the source must be named: "Which vulnerabilities affect Log4j?" asks
    # for CVEs, and guessing a group to expand from would bury them.
    infer_from: str | None
    # Words removed from the question before guessing the source; defaults to
    # whatever `intent` matched.
    wording: re.Pattern | None = None


RELATIONS = (
    Relation("mitigation", MITIGATION_INTENT, {"technique": "mitigated_by"}, "technique"),
    Relation(
        "group",
        GROUP_INTENT,
        {"technique": "used_by_groups", "cve": "exploited_by"},
        "technique",
        GROUP_WORDING,
    ),
    Relation("cve", CVE_INTENT, {"group": "cves"}, None),
)

FILTERS = """
           AND (%(doc_type)s::text IS NULL OR doc_type = %(doc_type)s)
           AND (%(platform)s::text IS NULL OR metadata->'platforms' ? %(platform)s)"""

ARM_SQL = {
    "id": f"""
  SELECT id, 'id' AS arm, row_number() OVER (ORDER BY source_id) AS rank, %(pin)s AS pinned
    FROM chunks
   WHERE embedding_model = %(model)s AND source_id = ANY(%(ids)s::text[]){FILTERS}""",
    "fts": f"""
  SELECT id, 'fts' AS arm, row_number() OVER (ORDER BY r DESC, id) AS rank, false AS pinned
    FROM (SELECT id, ts_rank_cd(tsv, tq) AS r
            FROM chunks,
                 (SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')::tsquery AS tq) q
           WHERE embedding_model = %(model)s AND tsv @@ tq{FILTERS}
           ORDER BY r DESC, id
           LIMIT %(depth)s) s""",
    "vector": f"""
  SELECT id, 'vector' AS arm, row_number() OVER (ORDER BY d, id) AS rank, false AS pinned
    FROM (SELECT id, embedding <=> %(q)s AS d
            FROM chunks
           WHERE embedding_model = %(model)s{FILTERS}
           ORDER BY d, id
           LIMIT %(depth)s) s""",
}
# Ties are broken by id in every arm: ts_rank_cd ties often, and an untied
# ORDER BY ... LIMIT lets PostgreSQL pick which tied rows make the cut, so the
# same question could rank differently from one run to the next.

# HNSW search explores hnsw.ef_search candidates (default 40) and can return no
# more than that, which would silently cap the vector arm below DEPTH.
EF_SEARCH = 2 * DEPTH

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
 ORDER BY f.pinned DESC, f.score DESC, {tiebreak}c.source_id
 LIMIT %(k)s
"""
# Exact score ties are common under RRF: ranks (1, 2) and (2, 1) in two arms sum
# to the same score. With a vector arm, the tie goes to the chunk closer to the
# question rather than to the alphabet, which once picked T1556.006 over T1621
# as the technique to expand "MFA push notification spam" from.
VECTOR_TIEBREAK = "c.embedding <=> %(q)s, "

# Target chunks listed under `key` in the given source chunks, closest to the
# question first.
EXPAND = """
SELECT m.source_id, m.doc_type, m.metadata->>'name' AS name,
       {similarity} AS similarity, false AS pinned,
       ARRAY['expand:' || t.source_id] AS arms, m.content
  FROM chunks t
  JOIN chunks m ON m.embedding_model = t.embedding_model
               AND m.doc_type = %(target)s
               AND (t.metadata -> %(key)s) ? m.source_id
 WHERE t.embedding_model = %(model)s
   AND t.doc_type = %(source_type)s
   AND t.source_id = ANY(%(sources)s::text[])
 ORDER BY {order}
"""


def extract_ids(question: str) -> list[str]:
    """IDs in the order they appear, upper-cased and de-duplicated."""
    return list(dict.fromkeys(m.upper() for m in ID_PATTERN.findall(question)))


def _doc_type_of(source_id: str) -> str | None:
    if source_id.startswith("CVE-"):
        return "cve"
    return {"T": "technique", "M": "mitigation", "G": "group"}.get(source_id[:1])


@lru_cache(maxsize=1)
def _group_names(model: str) -> tuple[tuple[re.Pattern, tuple[str, ...]], ...]:
    """(pattern, group IDs) for every group name and alias in the table.

    Cached per process, so a long-running app needs a restart after re-ingest.
    Single plain words ("Play", "Silence", "Snake", "Tick") match only with
    their own capitalisation, or "what role does X play" would find a
    ransomware group; names with spaces, digits or punctuation ("cozy bear",
    "apt29") match in any case. Two aliases are shared ("Thrip", "UAC-0056"),
    so a name can map to more than one group.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT source_id, metadata->>'name', metadata->'aliases'
                 FROM chunks WHERE doc_type = 'group' AND embedding_model = %s""",
            (model,),
        )
        for source_id, name, aliases in cur.fetchall():
            for alias in {name, *(aliases or [])}:
                owners[alias].add(source_id)
    patterns = []
    for alias, group_ids in owners.items():
        flags = 0 if re.fullmatch(r"[A-Za-z]+", alias) else re.IGNORECASE
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags)
        patterns.append((pattern, tuple(sorted(group_ids))))
    return tuple(patterns)


def named_groups(question: str) -> list[str]:
    """IDs of the groups a question names by name or alias."""
    found = {
        gid
        for pattern, group_ids in _group_names(config.EMBEDDING_MODEL)
        if pattern.search(question)
        for gid in group_ids
    }
    return sorted(found)


def search(
    question: str,
    k: int = 5,
    doc_type: str | None = None,
    platform: str | None = None,
    *,
    arms: Sequence[str] = ARMS,
    pin_ids: bool = False,
    expand: bool = False,
) -> list[dict]:
    """Ranked hits like query.search, with `similarity` holding the RRF score
    (cosine similarity for expanded chunks) and `arms` listing which arms found
    each chunk, at what rank."""
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {ARMS}")

    ids = extract_ids(question)
    named = ids + [g for g in named_groups(question) if g not in ids]
    lookup_ids = named if "id" in arms else ids
    active = [a for a in arms if a != "id" or lookup_ids]
    if not active:
        return []

    params = {
        "model": config.EMBEDDING_MODEL,
        "doc_type": doc_type,
        "platform": platform,
        "ids": lookup_ids,
        "text": question,
        "pin": pin_ids,
        "depth": DEPTH,
        "rrf_k": RRF_K,
        "k": k,
    }
    if "vector" in active:
        params["q"] = embed.encode_one(question)
    sql = SEARCH.format(
        ranked="\n UNION ALL".join(f"\n({ARM_SQL[a]})" for a in active),
        tiebreak=VECTOR_TIEBREAK if "vector" in active else "",
    )

    with db.connect() as conn, conn.cursor() as cur:
        db.ensure_model(cur)
        cur.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
        cur.execute(sql, params)
        columns = [c.name for c in cur.description]
        hits = [dict(zip(columns, row)) for row in cur.fetchall()]
        if expand and doc_type is None:
            hits = _expand(cur, sql, hits, named, params)
    return hits


def _expand(cur, sql: str, hits: list[dict], named: list[str], params: dict) -> list[dict]:
    """Insert the targets of every relation the question asks for after the
    pinned hits, in RELATIONS order."""
    expanded = []
    for relation in RELATIONS:
        if not relation.intent.search(params["text"]):
            continue
        for source_type, sources in _relation_sources(cur, sql, relation, named, params).items():
            expanded += _targets(cur, relation, source_type, sources, params)
    if not expanded:
        return hits

    merged, seen = [], set()
    for hit in [h for h in hits if h["pinned"]] + expanded + hits:
        if hit["source_id"] not in seen:
            merged.append(hit)
            seen.add(hit["source_id"])
    return merged[: params["k"]]


def _relation_sources(
    cur, sql: str, relation: Relation, named: list[str], params: dict
) -> dict[str, list[str]]:
    """Source chunk IDs to follow `relation` from, by doc type.

    These are the objects the question names, by ID or by group name, that are
    of a source type. A question naming only other objects ("What does
    mitigation M1032 recommend?", "Which group is known as Cozy Bear?") is about
    those and gets nothing. One naming nothing gets the top `infer_from` hit
    for the question with the relation's wording removed: restricting to that
    doc type matters because "mitigations" pulls generic mitigation chunks into
    the unrestricted top k; removing the wording matters because technique
    chunks carry mitigation guidance, so "mitigations reduce the risk of
    Kerberoasting" otherwise ranks Exploitation for Credential Access above
    Kerberoasting.
    """
    if named:
        by_type: dict[str, list[str]] = defaultdict(list)
        for source_id in named:
            source_type = _doc_type_of(source_id)
            if source_type in relation.keys:
                by_type[source_type].append(source_id)
        return by_type
    if relation.infer_from is None:
        return {}

    subject = (relation.wording or relation.intent).sub(" ", params["text"])
    source_params = {**params, "text": subject, "doc_type": relation.infer_from, "k": 1}
    if "q" in params:
        source_params["q"] = embed.encode_one(subject)
    cur.execute(sql, source_params)
    return {relation.infer_from: [row[0] for row in cur.fetchall()]}


def _targets(cur, relation: Relation, source_type: str, sources: list[str], params: dict) -> list[dict]:
    if "q" in params:
        sql = EXPAND.format(
            similarity="1 - (m.embedding <=> %(q)s)", order="m.embedding <=> %(q)s, m.source_id"
        )
    else:
        sql = EXPAND.format(similarity="0.0::float", order="m.source_id")
    cur.execute(
        sql,
        {
            **params,
            "target": relation.target,
            "key": relation.keys[source_type],
            "source_type": source_type,
            "sources": sources,
        },
    )
    columns = [c.name for c in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]
