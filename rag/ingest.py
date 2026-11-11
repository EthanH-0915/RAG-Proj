"""Chunk, embed and load ATT&CK techniques, mitigations and groups, and KEV
entries, into PostgreSQL.

Each doc type is a `build_*_chunk` function over the same table.

Usage:  python -m rag.ingest [--doc-types technique mitigation group cve] [--limit N] [--truncate]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from psycopg.types.json import Json

from . import config, db, embed, kev
from .attack import AttackData
from .text import clean_text

DOC_TYPE_TECHNIQUE = "technique"
DOC_TYPE_MITIGATION = "mitigation"
DOC_TYPE_GROUP = "group"
DOC_TYPE_CVE = "cve"


@dataclass
class Chunk:
    source_id: str
    doc_type: str
    content: str
    metadata: dict = field(default_factory=dict)
    # What the embedding model sees, when it should differ from `content`. The
    # model truncates at max_seq_length tokens, so long relationship lists would
    # be cut off anyway; they are matched by ID and keyword instead.
    embed_text: str | None = None


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
    embed_text = "\n".join(lines)

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
        # The relationship's own text says how this mitigation applies to this
        # technique, which is what an answer to "how do I defend against X" needs.
        lines += ["", "Mitigations:"]
        for mitigation in sorted(mitigations, key=lambda m: data.attack_id(m) or ""):
            mitigation_id = data.attack_id(mitigation)
            if not mitigation_id:
                continue
            guidance = data.relationship_text("mitigates", mitigation["id"], technique["id"])
            label = f"{mitigation_id} ({mitigation.get('name', '')})"
            lines.append(f"{label}: {guidance}" if guidance else label)
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
    return Chunk(attack_id, DOC_TYPE_TECHNIQUE, "\n".join(lines), metadata, embed_text)


def build_mitigation_chunk(data: AttackData, mitigation: dict) -> Chunk | None:
    attack_id = data.attack_id(mitigation)
    if not attack_id:
        return None

    techniques = data.related("mitigates", mitigation["id"], incoming=False)
    technique_ids = _plain_ids(data, techniques)

    lines = [f"{attack_id} — {mitigation.get('name', '')}"]
    description = clean_text(mitigation.get("description"))
    if description:
        lines += ["", "Description:", description]
    embed_text = "\n".join(lines)

    if technique_ids:
        # IDs only: each is a single lexeme, so "mitigations for T1110.003" finds
        # this chunk through fts, whereas the names of the 100+ techniques the
        # broadest mitigations cover would match nearly any keyword query.
        lines += [
            "",
            f"Mitigates ({len(technique_ids)} techniques): {', '.join(technique_ids)}",
        ]

    metadata = {
        "name": mitigation.get("name"),
        "mitigates": technique_ids,
        "attack_version": mitigation.get("x_mitre_version"),
        "url": data.attack_url(mitigation),
    }
    return Chunk(attack_id, DOC_TYPE_MITIGATION, "\n".join(lines), metadata, embed_text)


def build_group_chunk(data: AttackData, group: dict) -> Chunk | None:
    attack_id = data.attack_id(group)
    if not attack_id:
        return None

    name = group.get("name", "")
    aliases = [a for a in group.get("aliases", []) if a != name]
    used = data.related("uses", group["id"], incoming=False)
    techniques = [o for o in used if o["type"] == "attack-pattern"]
    software = [o for o in used if o["type"] in ("malware", "tool")]
    campaigns = data.related("attributed-to", group["id"], incoming=True)
    technique_ids = _plain_ids(data, techniques)
    cves = data.cves_exploited_by(group)

    # Aliases are embedded: "Cozy Bear" or "Midnight Blizzard" is often the only
    # name a question uses.
    lines = [f"{attack_id} — {name}"]
    if aliases:
        lines.append(f"Also known as: {', '.join(aliases)}")
    description = clean_text(group.get("description"))
    if description:
        lines += ["", "Description:", description]
    embed_text = "\n".join(lines)

    if technique_ids:
        # IDs only, as in mitigation chunks: up to 130 technique names would
        # match nearly any keyword query.
        lines += ["", f"Techniques used ({len(technique_ids)}): {', '.join(technique_ids)}"]
    if software:
        # Names kept: the lists are short and the names distinctive (Mimikatz).
        lines += [f"Software used: {'; '.join(_id_with_name(data, software))}"]
    if campaigns:
        lines += [f"Campaigns: {'; '.join(_id_with_name(data, campaigns))}"]
    if cves:
        lines += [f"Exploited vulnerabilities: {', '.join(cves)}"]

    metadata = {
        "name": name,
        "aliases": aliases,
        "techniques": technique_ids,
        "software": _plain_ids(data, software),
        "campaigns": _plain_ids(data, campaigns),
        "cves": cves,
        "attack_version": group.get("x_mitre_version"),
        "url": data.attack_url(group),
    }
    return Chunk(attack_id, DOC_TYPE_GROUP, "\n".join(lines), metadata, embed_text)


def build_cve_chunk(data: AttackData, entry: dict) -> Chunk | None:
    """One KEV entry, from kev.entries() so it carries `exploited_by`.

    Two fields stay out of `content`, which feeds the fts index. requiredAction
    is near-identical boilerplate ("Apply mitigations per vendor instructions
    ..."), so 1,700 copies would answer every "mitigations" keyword query; and
    the ransomware line is written only when use is Known, or every chunk would
    match "ransomware".
    """
    cve = entry.get("cveID")
    if not cve:
        return None

    lines = [
        f"{cve} — {entry.get('vulnerabilityName', '')}",
        f"Vendor: {entry.get('vendorProject', '')}; product: {entry.get('product', '')}",
    ]
    description = entry.get("shortDescription", "").strip()
    if description:
        lines += ["", "Description:", description]
    embed_text = "\n".join(lines)

    ransomware = entry.get("knownRansomwareCampaignUse") == "Known"
    lines.append("")
    if ransomware:
        lines.append("Known to be used in ransomware campaigns.")
    if entry.get("cwes"):
        lines.append(f"Weaknesses: {', '.join(entry['cwes'])}")
    lines.append(
        f"Added to KEV: {entry.get('dateAdded', '')} (federal due date {entry.get('dueDate', '')})"
    )
    exploiters = entry["exploited_by"]
    if exploiters:
        lines.append(f"Exploited by (ATT&CK): {'; '.join(_id_with_name(data, exploiters))}")

    metadata = {
        "name": entry.get("vulnerabilityName"),
        "vendor": entry.get("vendorProject"),
        "product": entry.get("product"),
        "date_added": entry.get("dateAdded"),
        "due_date": entry.get("dueDate"),
        "ransomware_use": ransomware,
        "cwes": entry.get("cwes", []),
        "required_action": entry.get("requiredAction"),
        "exploited_by": _plain_ids(data, exploiters),
        "references": [u.strip() for u in entry.get("notes", "").split(";") if u.strip()],
        "url": f"https://nvd.nist.gov/vuln/detail/{cve}",
    }
    return Chunk(cve, DOC_TYPE_CVE, "\n".join(lines), metadata, embed_text)


# doc_type -> (objects to chunk, chunk builder)
BUILDERS = {
    DOC_TYPE_TECHNIQUE: (AttackData.techniques, build_technique_chunk),
    DOC_TYPE_MITIGATION: (AttackData.mitigations, build_mitigation_chunk),
    DOC_TYPE_GROUP: (AttackData.groups, build_group_chunk),
    DOC_TYPE_CVE: (kev.entries, build_cve_chunk),
}


UPSERT = """
INSERT INTO chunks (source_id, doc_type, embedding_model, metadata, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (source_id, doc_type, embedding_model) DO UPDATE
   SET metadata  = EXCLUDED.metadata,
       content   = EXCLUDED.content,
       embedding = EXCLUDED.embedding
"""


def load(chunks: list[Chunk], truncate: bool) -> None:
    vectors = embed.encode([c.embed_text or c.content for c in chunks], progress=True)
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
    parser.add_argument(
        "--doc-types", nargs="+", choices=list(BUILDERS), default=list(BUILDERS),
        help="doc types to ingest (default: all)",
    )
    parser.add_argument("--limit", type=int, help="ingest only the first N of each doc type")
    parser.add_argument(
        "--truncate", action="store_true",
        help="empty the whole table, every doc type, before loading",
    )
    args = parser.parse_args()

    print(f"Reading {config.ATTACK_PATH.name} ...")
    data = AttackData(config.ATTACK_PATH)
    print(f"  ATT&CK v{data.collection.get('x_mitre_version')}")

    chunks: list[Chunk] = []
    for doc_type in args.doc_types:
        objects, build = BUILDERS[doc_type]
        selected = list(objects(data))
        if args.limit:
            selected = selected[: args.limit]
        built = [c for obj in selected if (c := build(data, obj))]
        print(f"  {len(built)} {doc_type} chunks")
        chunks += built

    sizes = sorted(len(c.content) for c in chunks)
    print(
        f"  content chars: min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
        f"max {sizes[-1]}"
    )

    limit = embed.max_tokens()
    counts = embed.token_counts([c.embed_text or c.content for c in chunks])
    over = [c.source_id for c, n in zip(chunks, counts) if n > limit]
    print(f"  embed tokens: median {sorted(counts)[len(counts) // 2]}, max {max(counts)}")
    if over:
        print(
            f"  warning: {len(over)}/{len(chunks)} embed texts exceed the model's "
            f"{limit}-token limit and lose their tail, e.g. {', '.join(over[:5])}"
        )

    print(f"Embedding with {config.EMBEDDING_MODEL} ...")
    load(chunks, args.truncate)
    print(f"Loaded {len(chunks)} chunks into {config.DATABASE_URL.rsplit('@', 1)[-1]}")


if __name__ == "__main__":
    main()
