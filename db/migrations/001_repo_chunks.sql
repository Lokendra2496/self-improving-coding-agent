CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repo_chunks (
    id UUID PRIMARY KEY,
    source_path TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS repo_chunks_embedding_idx
ON repo_chunks USING ivfflat (embedding vector_cosine_ops);
