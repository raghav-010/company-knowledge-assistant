"""
utils.py — shared engine + vector store for Company Knowledge Assistant
Raghav Balaji V | github.com/raghav-010

Design choice: HuggingFaceEmbeddings (all-MiniLM-L6-v2) instead of OpenAI
text-embedding-3-small. Reason: OpenAI requires international USD card
payment which doesn't work with Indian cards. HuggingFace runs the model
locally inside Docker — completely free, no API key needed.
Trade-off: 384-dim vectors vs 1536-dim, acceptable for this project size.
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore

PG_CONN_STR = os.getenv("DATABASE_URL")

PG_ENGINE = PGEngine.from_connection_string(PG_CONN_STR)

# Local embedding model — runs inside the Docker container, zero cost
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


async def get_vector_store() -> AsyncPGVectorStore:
    return await AsyncPGVectorStore.create(
        engine=PG_ENGINE,
        embedding_service=embeddings,
        table_name="langchain_pg_embedding",
        metadata_json_column="langchain_metadata",
    )
