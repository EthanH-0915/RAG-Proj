"""Chunk, embed and load ATT&CK techniques into PostgreSQL.

Week 3 scope: techniques and sub-techniques only. Groups, mitigations and KEV
entries are additional `build_*_chunks` functions over the same table, added
once the end-to-end path is working.

Usage:  python -m rag.ingest [--limit N] [--truncate]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from psycopg.types.json import Json

from . import config, db, embed
from .attack import AttackData
from .text import clean_text

DOC_TYPE_TECHNIQUE = "technique"


@dataclass
class Chunk:
    source_id: str
    doc_type: str
    content: str
    metadata: dict = field(default_factory=dict)


def _id_with_name(data: AttackData, objects: list[dict]) -> list[str]:
    """['M1017 (User Training)', ...] for the human-readable lines in `content`."""
    out = []
    for obj in objects:
        attack_id = data.attack_id(obj)
        if attack_id:
            out.append(f"{attack_id} ({obj.get('name', '')})")
    return sorted(out)


def _plain_ids(data: AttackData, objects: list[dict]) -> list[str]:
    return sorted({i for i in (data.attack_id(o) for o in objects) if i})


def build_technique_chunk(data: AttackData, technique: dict) -> Chunk | None:
    attack_id = data.attack_id(technique)
    if not attack_id:
        return None

    tactics = data.tactics_of(technique)
    platforms = technique.get("x_mitre_platforms", [])
    parent = data.parent_of(technique)
    mitigations = data.related("mitigates", technique["id"], incoming=True)

    users = data.related("uses", technique["id"], incoming=True)
    groups = [o for o in users if o["type"] == "intrusion-set"]
    software = [o for o in users if o["type"] in ("malware", "tool")]
    campaigns = [o for o in users if o["type"] == "campaign"]

    strategies = data.detection_strategies(technique)

    lines = [f"{attack_id} — {technique.get('name', '')}"]
    if parent:
        lines.append(
            f"Sub-technique of: {data.attack_id(parent)} ({parent.get('name', '')})"
        )
    if tactics:
        lines.append(f"Tactics: {', '.join(tactics)}")
    if platforms:
        lines.append(f"Platforms: {', '.join(platforms)}")

    description = clean_text(technique.get("description"))
    if description:
        lines += ["", "Description:", description]

    if strategies:
        lines += ["", "Detection:"]
        for strategy in strategies:
            lines.append(f"{strategy['id']} — {strategy['name']}")
            for analytic in strategy["analytics"]:
                platform = (
                    f" [{', '.join(analytic['platforms'])}]"
                    if analytic["platforms"]
                    else ""
                )
                lines.append(f"  {analytic['id']}{platform}: {analytic['description']}")

    if mitigations:
        lines += ["", f"Mitigations: {'; '.join(_id_with_name(data, mitigations))}"]
    if groups:
        lines += ["", f"Used by groups: {'; '.join(_id_with_name(data, groups))}"]
    if software:
        lines += [f"Used by software: {'; '.join(_id_with_name(data, software))}"]

    metadata = {
        "name": technique.get("name"),
        "tactics": tactics,
        "tactic_shortnames": [
            p["phase_name"]
            for p in technique.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"
        ],
        "platforms": platforms,
        "is_subtechnique": bool(technique.get("x_mitre_is_subtechnique")),
        "parent_id": data.attack_id(parent) if parent else None,
        # Relationship IDs live here so relational questions can be retrieved and
        # scored by ID instead of by matching an ID inside a paragraph of prose.
        "mitigated_by": _plain_ids(data, mitigations),
        "used_by_groups": _plain_ids(data, groups),
        "used_by_software": _plain_ids(data, software),
        "used_in_campaigns": _plain_ids(data, campaigns),
        "detection_strategies": [s["id"] for s in strategies],
        "attack_version": technique.get("x_mitre_version"),
        "url": data.attack_url(technique),
    }
    return Chunk(attack_id, DOC_TYPE_TECHNIQUE, "\n".join(lines), metadata)


UPSERT = """
INSERT INTO chunks (source_id, doc_type, embedding_model, metadata, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (source_id, doc_type, embedding_model) DO UPDATE
   SET metadata  = EXCLUDED.metadata,
       content   = EXCLUDED.content,
       embedding = EXCLUDED.embedding
"""


def load(chunks: list[Chunk], truncate: bool) -> None:
    vectors = embed.encode([c.content for c in chunks], progress=True)
    manifest = (
        json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
        if config.MANIFEST_PATH.exists()
        else {}
    )

    with db.connect() as conn, conn.cursor() as cur:
        if truncate:
            cur.execute("TRUNCATE chunks RESTART IDENTITY")
        cur.executemany(
            UPSERT,
            [
                (
                    c.source_id,
                    c.doc_type,
                    config.EMBEDDING_MODEL,
                    Json(c.metadata),
                    c.content,
                    vectors[i],
                )
                for i, c in enumerate(chunks)
            ],
        )
        cur.execute(
            """INSERT INTO ingest_runs (embedding_model, doc_types, n_chunks, manifest)
               VALUES (%s, %s, %s, %s)""",
            (
                config.EMBEDDING_MODEL,
                sorted({c.doc_type for c in chunks}),
                len(chunks),
                Json(manifest),
            ),
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="ingest only the first N techniques")
    parser.add_argument(
        "--truncate", action="store_true", help="empty the table before loading"
    )
    args = parser.parse_args()

    print(f"Reading {config.ATTACK_PATH.name} ...")
    data = AttackData(config.ATTACK_PATH)
    print(f"  ATT&CK v{data.collection.get('x_mitre_version')}")

    techniques = list(data.techniques())
    if args.limit:
        techniques = techniques[: args.limit]

    chunks = [c for t in techniques if (c := build_technique_chunk(data, t))]
    sizes = sorted(len(c.content) for c in chunks)
    print(f"  {len(chunks)} technique chunks")
    print(
        f"  content chars: min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
        f"max {sizes[-1]}"
    )

    print(f"Embedding with {config.EMBEDDING_MODEL} ...")
    load(chunks, args.truncate)
    print(f"Loaded {len(chunks)} chunks into {config.DATABASE_URL.rsplit('@', 1)[-1]}")


if __name__ == "__main__":
    main()
