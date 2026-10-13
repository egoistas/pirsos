CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY,
  doc_key text NOT NULL,
  version_number integer NOT NULL,
  filename text NOT NULL,
  sha256 char(64) UNIQUE NOT NULL,
  stored_path text NOT NULL,
  uploaded_at timestamptz NOT NULL DEFAULT now(),
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','processing','ready','failed')),
  is_active boolean NOT NULL DEFAULT false,
  pages integer,
  chunk_count integer NOT NULL DEFAULT 0,
  progress_done integer NOT NULL DEFAULT 0,
  progress_total integer NOT NULL DEFAULT 0,
  embed_model text NOT NULL,
  pipeline_version text NOT NULL,
  error_message text,
  UNIQUE(doc_key, version_number)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_document_per_key ON documents (doc_key) WHERE is_active;
CREATE INDEX IF NOT EXISTS documents_doc_key_idx ON documents(doc_key);

CREATE TABLE IF NOT EXISTS chunks (
  id bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal integer NOT NULL,
  group_key text NOT NULL,
  breadcrumb text NOT NULL,
  section text NOT NULL DEFAULT '',
  paragraph text NOT NULL DEFAULT '',
  subsection text NOT NULL DEFAULT '',
  page_start integer NOT NULL,
  page_end integer NOT NULL,
  kind text NOT NULL,
  body text NOT NULL,
  search_text text NOT NULL,
  content_sha256 char(64) NOT NULL,
  embed_model text NOT NULL,
  embedding vector(1024) NOT NULL,
  UNIQUE(document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS chunks_vector_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx ON chunks USING gist (search_text gist_trgm_ops);
CREATE INDEX IF NOT EXISTS chunks_group_idx ON chunks(document_id,group_key,ordinal);

CREATE TABLE IF NOT EXISTS embedding_cache (
  embed_model text NOT NULL,
  content_sha256 char(64) NOT NULL,
  embedding vector(1024) NOT NULL,
  PRIMARY KEY(embed_model, content_sha256)
);

CREATE TABLE IF NOT EXISTS document_events (
  id bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  happened_at timestamptz DEFAULT now(),
  actor text NOT NULL,
  event text NOT NULL,
  detail text NOT NULL DEFAULT ''
);
