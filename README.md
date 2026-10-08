# Company Knowledge Assistant (CKA)

A production-style **Retrieval-Augmented Generation (RAG)** system that lets employees ask natural-language questions about company documents and get grounded, cited answers.

> Built by **Raghav Balaji V** as Project #9 in my data science journey (April 2026 → present).  
> Portfolio: [raghav-010.github.io](https://raghav-010.github.io) | LinkedIn: [raghavbalaji010](https://linkedin.com/in/raghavbalaji010)

---

## What it does

Upload company documents (PDFs, Word docs, Markdown, text files) and ask questions like:

- *"How many PTO days can I carry forward?"*
- *"What is the daily meal reimbursement limit when traveling internationally?"*
- *"What is the remote work policy?"*

The system retrieves the most relevant chunks from the document store, reranks them for precision, and generates a grounded answer — refusing to hallucinate if the answer isn't in the docs.

---

## Architecture

```
User Question
      │
      ▼
FastAPI /ask endpoint
      │
      ▼
pgvector similarity search (top-5 chunks)
[HuggingFace all-MiniLM-L6-v2 embeddings, 384-dim]
      │
      ▼
Cohere Reranking (rerank-multilingual-v3.0)
→ keeps top-3 most relevant chunks
      │
      ▼
Gemini 2.5 Flash (LLM)
→ generates answer grounded in context
→ says "I don't know" if answer not in docs
      │
      ▼
Response: { answer, sources, contexts }
```

**Stack:**

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Runs locally in Docker, 100% free. OpenAI embeddings require international USD card payment not available with Indian cards. 384-dim vectors, sufficient for this project. |
| Vector DB | PostgreSQL + pgvector | Already in the stack — no extra service. HNSW index for fast approximate nearest-neighbour search. |
| Reranking | Cohere `rerank-multilingual-v3.0` | Cross-encoder re-scores top-5 chunks by semantic relevance → keeps top-3. Improves answer precision significantly over raw cosine similarity. |
| LLM | Google Gemini 2.5 Flash | Free tier via Google AI Studio. Replaced GPT-4o-mini (OpenAI account needed USD top-up, Indian card unsupported). Same quality for this use case. |
| API | FastAPI + Uvicorn | Async, lightweight, auto-generates `/docs` UI. |
| Observability | LangSmith | Traces every run: question, retrieved chunks, prompt, answer, latency. |
| Evaluation | RAGAS | Measures faithfulness, answer relevancy, context precision, context recall. |
| Containerisation | Docker + Docker Compose | Reproducible environment, one-command startup. |

---

## Project structure

```
company-knowledge-assistant/
├── app/
│   ├── api.py          # FastAPI: GET /, POST /ingest, GET /ingest/status, POST /ask
│   ├── ingest.py       # Document loading → chunking → pgvector storage → HNSW index
│   ├── rag.py          # RAG chain: retrieval → Cohere reranking → Gemini answer
│   ├── utils.py        # Shared: PGEngine, HuggingFace embeddings, vector store
│   ├── eval_ragas.py   # RAGAS evaluation script (run separately)
│   └── static/         # Frontend UI (HTML + CSS)
├── data/               # Company documents (subfolders = categories)
│   ├── announcements/
│   ├── faqs/
│   ├── guides/
│   ├── handbooks/
│   └── policies/
├── init-db/
│   └── init.sql        # Creates pgvector extension + embeddings table (384-dim)
├── seed/
│   └── qna_test.json   # Test Q&A pairs for RAGAS evaluation
├── .env.example        # Environment variable template (copy to .env)
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick start

### Prerequisites
- Docker Desktop installed and running
- A Google AI Studio API key (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))
- A Cohere API key (free at [dashboard.cohere.com](https://dashboard.cohere.com))

### 1. Clone and configure

```bash
git clone https://github.com/raghav-010/company-knowledge-assistant.git
cd company-knowledge-assistant
cp .env.example .env
# Edit .env and fill in your GOOGLE_API_KEY and CO_API_KEY
```

### 2. Start containers

```bash
docker compose up --build -d
```

First build takes ~20 minutes (downloads Python packages and HuggingFace model weights).  
Subsequent starts take seconds.

**Wait for the app to be ready:**
```bash
docker compose logs app --tail=20
# Look for: "Application startup complete."
```

### 3. Ingest documents

Open [http://localhost:8000](http://localhost:8000) and click **Ingest Data**.  
Or via API:
```bash
curl -X POST http://localhost:8000/ingest
```

Check progress:
```bash
curl http://localhost:8000/ingest/status
```

### 4. Ask questions

Via the UI at [http://localhost:8000](http://localhost:8000), or:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the daily meal reimbursement limit for international travel?"}'
```

### 5. Stop

```bash
docker compose down          # Stops containers, data preserved
docker compose down -v       # Stops + DELETES all ingested data
```

---

## Adding your own documents

Drop files into the appropriate `data/` subfolder:

```
data/policies/     → HR policies, expense policies
data/guides/       → How-to guides, setup instructions
data/handbooks/    → Employee handbook
data/faqs/         → FAQ documents
data/announcements/ → Company announcements
```

Supported formats: `.pdf`, `.docx`, `.md`, `.txt`

Then re-ingest via the UI or `POST /ingest`.

---

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/ingest` | POST | Start document ingestion |
| `/ingest/status` | GET | Check ingest progress |
| `/ask` | POST | Ask a question |
| `/docs` | GET | Auto-generated FastAPI docs |

**POST /ask request body:**
```json
{ "question": "How many PTO days can I carry forward?" }
```

**POST /ask response:**
```json
{
  "answer": "You can carry forward a maximum of 6 PTO days to the next year.",
  "sources": ["data/policies/PTO Policy.pdf"],
  "contexts": ["...relevant chunk text..."]
}
```

---

## Running RAGAS evaluation

With Docker running and documents ingested:

```bash
docker compose exec app python -m app.eval_ragas
```

Outputs 4 metrics averaged across the test set in `seed/qna_test.json`:
- **faithfulness** — answer grounded in retrieved context?
- **answer_relevancy** — answer addresses the question?
- **context_precision** — retrieved chunks relevant to the question?
- **context_recall** — retrieved context covers the reference answer?

---

## Key design decisions

**Why HuggingFace instead of OpenAI embeddings?**  
OpenAI API requires a credit card that supports international USD payments. My Indian card does not support this. `all-MiniLM-L6-v2` runs locally inside the Docker container — completely free, no API key needed. Trade-off: 384-dim vectors instead of 1536-dim, which is fine for this project's document size.

**Why Gemini instead of GPT-4o-mini?**  
OpenAI account had zero credits and Indian card cannot top up in USD. Gemini free tier works with any Google account via AI Studio. Same response quality for grounded Q&A.

**Why Cohere reranking?**  
pgvector's cosine similarity returns the top-K most similar chunks by vector distance — but "similar" isn't always "relevant." A Cohere cross-encoder re-scores them by actual semantic relevance to the question and keeps only the top-3. This measurably improves answer quality.

**Why HNSW index?**  
Brute-force vector search is O(n). HNSW gives O(log n) approximate nearest-neighbour search with ~99% recall. At ingestion size this doesn't matter, but it's the production-correct choice.

**Why PostgreSQL + pgvector instead of Pinecone/Chroma?**  
pgvector is already part of the stack (Docker Compose). No extra managed service, no extra cost, no extra API key. Same vector search capability for this scale.

---

## What's next

- [ ] Add Redis semantic caching (reduces repeat query cost and latency)
- [ ] Implement metadata filtering UI (let users select document category)
- [ ] Swap in personal documents (certificates, notes) and re-ingest
- [ ] Deploy to a live URL (Railway / Render / fly.io)

---

## Related projects

- [Customer Churn Prediction](https://github.com/raghav-010/churn-prediction) — Live Streamlit ML app
- [Retail Customer Behaviour Insights](https://github.com/raghav-010/retail-behaviour-insights) — Python + PostgreSQL + Power BI
- [Zoho Invoice Classifier](https://github.com/raghav-010/zoho-invoice-classifier) — ETL automation

---

*Built as part of my learning journey: Data Analyst → Data Scientist → ML Engineer → AI Engineer → own AI startup (2028+)*
