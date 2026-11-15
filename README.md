# RAG-Proj

A threat knowledge assistant over MITRE ATT&CK and CISA's Known Exploited
Vulnerabilities catalog, answering questions with a source ID cited for every
claim. Retrieval is PostgreSQL + pgvector; generation is a local LLM.

Research question: does hybrid retrieval (exact ID lookup + keyword + vector)
beat pure vector search, and for which kinds of question?

## Quickstart

```bash
docker compose up -d                 # PostgreSQL 17 + pgvector on localhost:5433
cp .env.example .env

python -m venv .venv
.venv/Scripts/activate               # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m rag.download               # caches ATT&CK + KEV into data/raw/ with a manifest
python -m rag.ingest --truncate      # chunk, embed and load techniques, mitigations, groups and KEV
python -m rag.query "How do attackers dump credentials from memory?"
python -m rag.query "What is T1059.001?" --mode hybrid
python -m rag.evaluate               # recall@5 and MRR@5 over eval/questions.yaml
python -m rag.evaluate --mode all    # every retrieval mode side by side
```

`sql/schema.sql` is applied only when the database volume is first created. To
pick up a schema change: `docker compose down -v && docker compose up -d`.

## Layout

| Path | Role |
| --- | --- |
| `sql/schema.sql` | The `chunks` table, its four indexes, and `ingest_runs` |
| `rag/config.py` | Paths, feed URLs, database URL, embedding model |
| `rag/download.py` | Caches the raw feeds; writes `data/raw/manifest.json` |
| `rag/attack.py` | STIX reader: techniques, mitigations, groups, relationships, detection strategies |
| `rag/kev.py` | KEV reader, joined to the ATT&CK groups and software whose text names each CVE |
| `rag/ingest.py` | Builds chunks, embeds them, upserts into PostgreSQL |
| `rag/query.py` | Vector search, the `RETRIEVERS` mode registry, and the search CLI |
| `rag/hybrid.py` | ID and group-name lookup + full-text + vector, fused with RRF, plus the `RELATIONS` link-following table (technique → mitigations, technique/CVE → groups, group → CVEs); `fts`, `vector+fts`, `hybrid`, `hybrid-pin`, `hybrid+expand` |
| `rag/evaluate.py` | Scores retrieval against the question set; `--out` saves JSON with feed versions |
| `eval/questions.yaml` | Seed evaluation questions with expected source IDs |

## Status

All four doc types (techniques and sub-techniques, mitigations, groups and KEV
entries) are chunked, embedded and searchable by vector similarity or hybrid
retrieval with RRF, and `eval/questions.yaml` holds 59 questions. Still to come
are the LLM generation step and the Streamlit UI.

## Data provenance

`data/raw/manifest.json` records the URL, fetch time, SHA-256 and version of
each feed, and `rag/ingest.py` copies it into the `ingest_runs` table. Results
tables in the report should quote the ATT&CK version and KEV catalog version
from there, since both feeds change under you — KEV almost daily.
