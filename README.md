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
python -m rag.ingest --truncate      # chunk, embed and load techniques
python -m rag.query "How do attackers dump credentials from memory?"
```

`sql/schema.sql` is applied only when the database volume is first created. To
pick up a schema change: `docker compose down -v && docker compose up -d`.

## Layout

| Path | Role |
| --- | --- |
| `sql/schema.sql` | The `chunks` table, its four indexes, and `ingest_runs` |
| `rag/config.py` | Paths, feed URLs, database URL, embedding model |
| `rag/download.py` | Caches the raw feeds; writes `data/raw/manifest.json` |
| `rag/attack.py` | STIX reader: techniques, relationships, detection strategies |
| `rag/ingest.py` | Builds chunks, embeds them, upserts into PostgreSQL |
| `rag/query.py` | Pure vector search from the command line |
| `eval/questions.yaml` | Seed evaluation questions with expected source IDs |

## Status

Week 3 scope is in place for one doc type: techniques and sub-techniques are
chunked, embedded and searchable by vector similarity. Still to come are the
`group`, `mitigation` and `cve` chunk builders, hybrid retrieval with RRF, the
evaluation harness, the LLM generation step and the Streamlit UI.

## Data provenance

`data/raw/manifest.json` records the URL, fetch time, SHA-256 and version of
each feed, and `rag/ingest.py` copies it into the `ingest_runs` table. Results
tables in the report should quote the ATT&CK version and KEV catalog version
from there, since both feeds change under you — KEV almost daily.
