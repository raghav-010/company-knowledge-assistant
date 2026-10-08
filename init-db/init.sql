-- Enable the pgvector extension for storing and querying vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Main embeddings table
-- embedding vector(384): HuggingFace all-MiniLM-L6-v2 produces 384-dim vectors
-- (instructor uses 1536-dim for OpenAI; we use 384-dim for HuggingFace)
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    langchain_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content          text,
    embedding        vector(384),
    langchain_metadata jsonb,
    category         text
);
