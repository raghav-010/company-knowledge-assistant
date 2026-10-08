# Company Knowledge Assistant

A production-style **Retrieval-Augmented Generation (RAG)** system that answers employee questions grounded strictly in company documents — with cited sources and zero hallucination.

Built with LangChain, HuggingFace, Gemini, Cohere, PostgreSQL + pgvector, FastAPI, and Docker.

**Live demo:** Run locally with one command (see Quick Start below).  
**Author:** Raghav Balaji V — [raghav-010.github.io](https://raghav-010.github.io) · [LinkedIn](https://linkedin.com/in/raghavbalaji010) · [GitHub](https://github.com/raghav-010)

---

## Problem

Employees waste time searching through PDFs, handbooks, and policy documents to find answers to routine questions. This system ingests those documents once and lets anyone ask questions in plain English — returning precise, sourced answers in under 10 seconds.

---

## How it works

```
User Question
      │
      ▼
FastAPI /ask endpoint
      │
      ▼
pgvector similarity search
HuggingFace all-MiniLM-L6-v2 embeddings · 384-dim · HNSW index
Retrieves top-5 semantically similar chunks
      │
      ▼
Cohere Reranking (rerank-multilingual-v3.0)
Cross-encoder re-scores chunks by relevance → keeps top-3
      │
      ▼
Google Gemini (LLM)
Generates grounded answer from context
Returns "I don't know" if answer not in documents
      │
      ▼
{ answer, sources, contexts }
```

---

## Stack

| Layer | Technology | Decision |
|-------|-----------|----------|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Runs locally inside Docker — no API cost, no external dependency. 384-dim vectors. |
| Vector store | PostgreSQL + pgvector | Single stack, no extra managed service. HNSW index for O(log n) ANN search. |
| Reranking | Cohere `rerank-multilingual-v3.0` | Cross-encoder precision over cosine similarity. Top-5 → top-3 by true relevance. |
| LLM | Google Gemini Flash | Fast, free tier via Google AI Studio. Grounded generation with strict system prompt. |
| API | FastAPI + Uvicorn | Async, auto-documented, production-ready. |
| Observability | LangSmith | Full trace per request: question, chunks, prompt, answer, latency. |
| Evaluation | RAGAS | Faithfulness, answer relevancy, context precision, context recall. |
| Infrastructure | Docker + Docker Compose | One-command reproducible environment. |

---

## Features

- **Multi-format ingestion** — PDF, DOCX, Markdown, plain text
- **Category-aware retrieval** — documents organised by subfolder (policies, faqs, guides, handbooks, announcements)
- **Cohere reranking** — improves answer quality beyond raw vector similarity
- **Hallucination guard** — LLM instructed to say "I don't know" if answer not in context
- **Source citations** — every answer shows which document it came from
- **HNSW index** — approximate nearest-neighbour search, production-correct choice
- **LangSmith tracing** — full observability on every query
- **RAGAS evaluation** — automated quality scoring on a test set
- **Clean UI** — Lucent interface, dark theme, runs in the browser at `localhost:8000`

---

## Project structure

```
company-knowledge-assistant/
├── app/
│   ├── api.py          # FastAPI: GET /, POST /ingest, GET /ingest/status, POST /ask
│   ├── ingest.py       # Document loading → chunking → pgvector → HNSW index
│   ├── rag.py          # Retrieval → Cohere reranking → Gemini answer generation
│   ├── utils.py        # PGEngine, HuggingFace embeddings, vector store
│   ├── eval_ragas.py   # RAGAS evaluation (run separately)
│   └── static/         # Lucent UI (HTML + CSS)
├── data/
│   ├── announcements/
│   ├── faqs/
│   ├── guides/
│   ├── handbooks/
│   └── policies/
├── init-db/
│   └── init.sql        # pgvector extension + embeddings table (vector 384)
├── seed/
│   └── qna_test.json   # Test Q&A pairs for RAGAS evaluation
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quick start

### Prerequisites
- Docker Desktop
- Google AI Studio API key (free — [aistudio.google.com/apikey](https://aistudio.google.com/apikey))
- Cohere API key (free — [dashboard.cohere.com](https://dashboard.cohere.com))

### 1. Clone and configure

```bash
git clone https://github.com/raghav-010/company-knowledge-assistant.git
cd company-knowledge-assistant
cp .env.example .env
# Open .env and fill in GOOGLE_API_KEY and CO_API_KEY
```

### 2. Start

```bash
docker compose up --build -d
```

First build downloads model weights (~20 min one-time). Subsequent starts take seconds.

```bash
docker compose logs app --tail=20
# Wait for: "Application startup complete."
```

### 3. Ingest your documents

Open [http://localhost:8000](http://localhost:8000) and click **Ingest Data**.

### 4. Ask questions

Use the UI at [http://localhost:8000](http://localhost:8000) or via API:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the daily meal reimbursement limit for international travel?"}'
```

**Response:**
```json
{
  "answer": "The daily meal reimbursement limit for international travel is US $60 per day (receipts required).",
  "sources": ["data/faqs/travel-faq.txt"],
  "contexts": ["...retrieved passage..."]
}
```

### 5. Stop

```bash
docker compose down        # Stops containers, data preserved
docker compose down -v     # Stops and deletes all ingested data
```

---

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Lucent UI |
| `/ingest` | POST | Start document ingestion (async background task) |
| `/ingest/status` | GET | Poll ingestion progress |
| `/ask` | POST | Ask a natural-language question |
| `/docs` | GET | Auto-generated FastAPI interactive docs |

---

## Adding your own documents

Drop files into the appropriate `data/` subfolder and re-ingest:

```
data/policies/      → HR policies, expense policies
data/guides/        → How-to guides, setup docs
data/handbooks/     → Employee handbook
data/faqs/          → FAQ documents
data/announcements/ → Company announcements
```

Supported: `.pdf` `.docx` `.md` `.txt`

---

## Running RAGAS evaluation

```bash
docker compose exec app python -m app.eval_ragas
```

Outputs four metrics averaged across `seed/qna_test.json`:

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer relevancy | Does the answer address the question asked? |
| Context precision | Are the retrieved chunks relevant to the question? |
| Context recall | Does the context cover the reference answer? |

---

## Design decisions

**HuggingFace over OpenAI embeddings** — `all-MiniLM-L6-v2` runs entirely inside the Docker container with no external API call and no cost. 384-dim vectors are sufficient for this document scale.

**Cohere reranking** — vector cosine similarity finds chunks that are broadly similar; a cross-encoder reranker finds chunks that are specifically relevant to the question. The two-stage approach (retrieve 5, rerank to 3) measurably improves answer quality.

**PostgreSQL + pgvector over Pinecone or Chroma** — pgvector is already part of the Docker Compose stack. No extra managed service, no extra API key, same vector search capability at this scale.

**HNSW index** — O(log n) approximate nearest-neighbour search versus brute-force O(n). Production-correct choice from day one.

**Strict system prompt** — the LLM is instructed to answer only from provided context and return "I don't know" otherwise. This eliminates hallucination at the cost of occasional non-answers when relevant content is missing from the document store.

---

## Roadmap

- [ ] Redis semantic caching — reduce latency on repeated questions
- [ ] Metadata filtering UI — let users scope queries to a specific document category
- [ ] Public deployment — Railway / Render / fly.io
- [ ] Swap in domain-specific documents and re-evaluate with RAGAS

---

## Related projects

- [Customer Churn Prediction](https://github.com/raghav-010/churn-prediction) — Live ML app (Streamlit, Scikit-learn, SMOTE)
- [Retail Customer Behaviour Insights](https://github.com/raghav-010/retail-behaviour-insights) — PostgreSQL + Power BI + RFM segmentation
- [Zoho Invoice Classifier](https://github.com/raghav-010/zoho-invoice-classifier) — Python ETL automation
