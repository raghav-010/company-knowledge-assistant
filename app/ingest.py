"""
ingest.py — Document ingestion pipeline for Company Knowledge Assistant
Raghav Balaji V | github.com/raghav-010

Pipeline:
  1. Load all docs from data/ subfolders (.pdf, .docx, .md, .txt)
  2. Set metadata["category"] = subfolder name (e.g. "policies", "guides")
  3. Chunk with RecursiveCharacterTextSplitter (900 chars, 120 overlap)
  4. Embed with HuggingFace (all-MiniLM-L6-v2) and store in pgvector
  5. Build HNSW index for fast approximate nearest-neighbour search
"""
from __future__ import annotations
import os, glob, asyncio, traceback
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy

DATA_DIR = os.getenv("DATA_DIR", "data")


def _load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []

    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue

        ext = os.path.splitext(path)[1].lower()
        relative_path = os.path.relpath(path, base)
        category = relative_path.split(os.sep)[0] if os.sep in relative_path else "general"

        try:
            loaded_docs: List[Document] = []
            if ext == ".md":
                loaded_docs = UnstructuredMarkdownLoader(path).load()
            elif ext == ".pdf":
                loaded_docs = PyMuPDFLoader(path).load()
            elif ext == ".docx":
                loaded_docs = UnstructuredWordDocumentLoader(path).load()
            elif ext == ".txt":
                loaded_docs = TextLoader(path).load()

            for d in loaded_docs:
                d.metadata["category"] = category
                docs.append(d)

        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs


def _chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
    )
    return splitter.split_documents(docs)


async def _create_index(store) -> None:
    try:
        index = HNSWIndex(
            name="hnsw_idx",
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
            m=16,
            ef_construction=64,
        )
        await store.aapply_vector_index(index, concurrently=True)
        print("INGEST: HNSW index created successfully")
    except Exception as e:
        if "already exists" in str(e):
            print("INGEST: HNSW index already exists, skipping.")
        else:
            raise


async def run_ingest_async() -> dict:
    """Main ingest entry point called by the API."""
    docs = _load_docs()
    chunks = _chunk(docs)
    store = await get_vector_store()
    await store.aadd_documents(chunks)
    print(f"INGEST: {len(docs)} documents → {len(chunks)} chunks stored in pgvector")
    await _create_index(store)
    return {"documents": len(docs), "chunks": len(chunks)}