"""Pure vector search over the chunks table, from the command line.

This is the `retrieval_mode = vector` arm of the main experiment. Hybrid search
(ID lookup + full-text + RRF) joins it in week 4 behind the same interface.

Usage:
    python -m rag.query "How do attackers dump credentials from memory?"
    python -m rag.query "What is T1059.001?" -k 3 --show-content
    python -m rag.query "credential dumping" --platform Windows --doc-type technique
"""

from __future__ import annotations

import argparse
import textwrap

from . import config, db, embed

SEARCH = """
SELECT source_id,
       doc_type,
       metadata->>'name' AS name,
       1 - (embedding <=> %(q)s) AS similarity,
       content
  FROM chunks
 WHERE embedding_model = %(model)s
   AND (%(doc_type)s::text IS NULL OR doc_type = %(doc_type)s)
   AND (%(platform)s::text IS NULL OR metadata->'platforms' ? %(platform)s)
 ORDER BY embedding <=> %(q)s
 LIMIT %(k)s
"""


def search(
    question: str,
    k: int = 5,
    doc_type: str | None = None,
    platform: str | None = None,
) -> list[dict]:
    vector = embed.encode_one(question)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT embedding_model, count(*) FROM chunks GROUP BY embedding_model"
        )
        present = dict(cur.fetchall())
        if config.EMBEDDING_MODEL not in present:
            raise SystemExit(
                f"No chunks embedded with {config.EMBEDDING_MODEL}.\n"
                f"Models in the table: {present or 'none — run python -m rag.ingest'}"
            )

        cur.execute(
            SEARCH,
            {
                "q": vector,
                "model": config.EMBEDDING_MODEL,
                "doc_type": doc_type,
                "platform": platform,
                "k": k,
            },
        )
        columns = [c.name for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="chunks to return (default 5)")
    parser.add_argument("--doc-type", help="technique | group | mitigation | cve")
    parser.add_argument("--platform", help="e.g. Windows, Linux, macOS")
    parser.add_argument(
        "--show-content", action="store_true", help="print the whole chunk"
    )
    args = parser.parse_args()

    hits = search(args.question, args.k, args.doc_type, args.platform)
    print(f"\nQ: {args.question}\n")
    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank}. {hit['source_id']:<14} {hit['similarity']:.3f}  "
            f"{hit['name']}  [{hit['doc_type']}]"
        )
        body = hit["content"] if args.show_content else hit["content"][:240]
        print(textwrap.indent(body.strip(), "      "))
        print()
    print("retrieved: " + ", ".join(h["source_id"] for h in hits))


if __name__ == "__main__":
    main()
