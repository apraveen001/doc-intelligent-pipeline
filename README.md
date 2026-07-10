# DocMind — Intelligent Document Pipeline

A RAG (Retrieval-Augmented Generation) pipeline that lets you upload PDF documents, ask questions about them, and get answers grounded in what's actually in the document — not hallucinated from thin air.

Built with FastAPI, ChromaDB, Ollama, and Gemini.

---

## What it does

You upload a PDF. The pipeline breaks it into 500-word chunks, converts each chunk into a vector embedding using a local Ollama model, and stores everything in ChromaDB. When you ask a question, the same embedding model converts your question into a vector, finds the most semantically similar chunks, and passes them to Gemini to generate a grounded answer — along with confidence scores for each source chunk.

No external embedding API calls. No per-token costs on ingestion. Just a clean local-first pipeline with a cloud LLM only at the answer generation step.

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| API | FastAPI | Async, typed, auto-generates docs at `/docs` |
| Embeddings | Ollama + `nomic-embed-text` | Runs locally, zero cost, 768d vectors |
| Vector DB | ChromaDB Cloud | Persistent storage, cosine similarity search |
| LLM | Gemini 2.0 Flash | Fast, free tier, handles large context well |
| PDF parsing | pdfplumber | Better than pypdf for multi-column layouts |
| HTTP client | httpx | Async-native, works cleanly with FastAPI |

---

## Project structure

```
doc-intelligent-pipeline/
├── app/
│   └── src/
│       ├── main.py              # FastAPI app — all endpoints live here
│       ├── static/
│       │   └── index.html       # Frontend UI
│       ├── core/
│       │   ├── extraction.py    # PDF parsing + chunking
│       │   ├── embedding.py     # Ollama embedding calls
│       │   ├── vector_store.py  # ChromaDB read/write
│       │   └── query_pipeline.py# End-to-end query flow
│       └── llm/
│           └── llm.py           # Gemini API call + system prompt
├── pyproject.toml
└── README.md
```

---

## Getting started

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed locally.

**1. Clone and install:**
```bash
git clone https://github.com/your-username/doc-intelligent-pipeline.git
cd doc-intelligent-pipeline
pip install -e .
```

**2. Pull the embedding model:**
```bash
ollama pull nomic-embed-text
```

**3. Set up your `.env` file in `app/src/`:**
```env
CHROMA_API_KEY=your-chromadb-api-key
CHROMA_TENANT=your-tenant-id
CHROMA_DATABASE=your-database-name
GEMINI_API_KEY=your-gemini-api-key
```

**4. Run:**
```bash
cd app/src
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` — the UI loads there.

---

## API endpoints

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/upload` | Accepts a PDF, runs the full ingestion pipeline |
| `GET` | `/query?q=...` | Embeds query, retrieves chunks, returns LLM answer |
| `GET` | `/documents` | Lists all documents currently in the knowledge base |
| `DELETE` | `/collection/clear` | Wipes the entire ChromaDB collection |

API docs available at `http://127.0.0.1:8000/docs`.

---

## How the pipeline works

```
Upload
  PDF → pdfplumber extracts text
      → split into 500-word chunks (50-word overlap)
      → each chunk embedded via nomic-embed-text (768d)
      → upserted into ChromaDB with metadata

Query
  question → embedded with same model
           → cosine similarity search → top 3 chunks
           → chunks + question sent to Gemini
           → answer returned with source citations + confidence scores
```

The 50-word overlap between chunks means context isn't lost at boundaries — a sentence that spans two chunks will still be retrievable from either side.

---

## Environment notes

- Ollama must be running locally (`ollama serve`) before starting the API
- ChromaDB is cloud-hosted — make sure your API key has write access to the collection
- Gemini free tier allows 15 requests/minute — space out queries during testing
- The `.env` file is gitignored — never commit it

---

## Screenshots

**Upload a document**
![Upload](screenshots/upload.png)

**Ask a question**
![Query](screenshots/query.png)


## What's next

Things worth adding if you want to take this further:

- **Streaming responses** — stream Gemini tokens back to the frontend instead of waiting for the full answer
- **Conversation memory** — let users ask follow-up questions with context from previous turns
- **Agentic retrieval with LangGraph** — let the LLM decide when to re-query with a rephrased question before answering
- **MCP server** — expose the pipeline as a tool that any MCP-compatible client can call
- **Swap embedding model** — replace Ollama with Google's `text-embedding-004` for cloud deployment