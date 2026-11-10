"""Search the chunks table from the command line.

`search` here is pure vector search, the `retrieval_mode = vector` arm of the
main experiment; the other modes in RETRIEVERS are hybrid ablations from
rag/hybrid.py behind the same interface.

Usage:
    python -m rag.query "How do attackers dump credentials from memory?"
    python -m rag.query "What is T1059.001?" -k 3 --show-content --mode hybrid
    python -m rag.query "credential dumping" --platform Windows --doc-type technique
"""

from __future__ import annotations

import argparse
import textwrap
from functools import partial

from . import config, db, embed, hybrid

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
        db.ensure_model(cur)
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


# retrieval_mode -> search(question, k, doc_type, platform), returning ranked hits
# with `source_id` and `similarity` (cosine for vector, the RRF score otherwise).
RETRIEVERS = {
    "vector": search,
    "fts": partial(hybrid.search, arms=("fts",)),
    "vector+fts": partial(hybrid.search, arms=("fts", "vector")),
    "hybrid": hybrid.search,
    "hybrid-pin": partial(hybrid.search, pin_ids=True),
    "hybrid+expand": partial(hybrid.search, pin_ids=True, expand=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="chunks to return (default 5)")
    parser.add_argument("--mode", choices=list(RETRIEVERS), default="vector")
    parser.add_argument("--doc-type", help="technique | group | mitigation | cve")
    parser.add_argument("--platform", help="e.g. Windows, Linux, macOS")
    parser.add_argument(
        "--show-content", action="store_true", help="print the whole chunk"
    )
    args = parser.parse_args()

    hits = RETRIEVERS[args.mode](args.question, args.k, args.doc_type, args.platform)
    print(f"\nQ: {args.question}  ({args.mode})\n")
    for rank, hit in enumerate(hits, start=1):
        arms = f"  {', '.join(hit['arms'])}" if "arms" in hit else ""
        print(
            f"{rank}. {hit['source_id']:<14} {hit['similarity']:.4f}  "
            f"{hit['name']}  [{hit['doc_type']}]{arms}"
        )
        body = hit["content"] if args.show_content else hit["content"][:240]
        print(textwrap.indent(body.strip(), "      "))
        print()
    print("retrieved: " + ", ".join(h["source_id"] for h in hits))


if __name__ == "__main__":
    main()
