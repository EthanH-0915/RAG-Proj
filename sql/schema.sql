-- One table holds every chunk type; the two pipelines (ingestion, query) meet here.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id              SERIAL PRIMARY KEY,
  source_id       TEXT NOT NULL,              -- 'T1566.001', 'G0016', 'CVE-2021-44228'
  doc_type        TEXT NOT NULL,              -- technique | group | mitigation | cve
  -- Recorded per row so a query embedded with a different model is caught
  -- instead of silently returning nonsense distances.
  embedding_model TEXT NOT NULL,
  -- Carries the relationship IDs (mitigated_by, used_by_groups, ...) as well as
  -- the filterable fields (platforms, vendor, product), so relational questions
  -- can be answered and scored by ID rather than by matching text.
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
  content         TEXT NOT NULL,
  embedding       vector(384),
  tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  -- Makes ingestion idempotent: re-running upserts rather than duplicating.
  UNIQUE (source_id, doc_type, embedding_model)
);

CREATE INDEX IF NOT EXISTS chunks_source_id_idx ON chunks (source_id);
CREATE INDEX IF NOT EXISTS chunks_doc_type_idx  ON chunks (doc_type);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx  ON chunks USING gin (metadata jsonb_path_ops);

-- Provenance for the report: which snapshot of ATT&CK/KEV produced a results table.
CREATE TABLE IF NOT EXISTS ingest_runs (
  id              SERIAL PRIMARY KEY,
  finished_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  embedding_model TEXT NOT NULL,
  doc_types       TEXT[] NOT NULL,
  n_chunks        INTEGER NOT NULL DEFAULT 0,
  manifest        JSONB NOT NULL DEFAULT '{}'::jsonb
);
