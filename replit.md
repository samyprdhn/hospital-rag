# RAG Document QA

A full-stack Retrieval-Augmented Generation (RAG) web app for querying uploaded documents using multiple LLM providers.

## Architecture

```
artifacts/
├── rag-app/           # Python FastAPI backend + HTML/JS frontend
│   ├── main.py        # FastAPI app with upload, query, delete endpoints
│   ├── static/
│   │   └── index.html # Single-page frontend (no frameworks)
│   ├── requirements.txt
│   └── chroma_db/     # Local persistent vector DB (auto-created)
├── api-server/        # Node.js Express server (unused placeholder)
└── mockup-sandbox/    # Canvas design sandbox
```

## Workflows

- **RAG Document QA** — `uvicorn main:app` on port 5000 (Python FastAPI)
- **artifacts/api-server: API Server** — Node.js Express on port 8080
- **artifacts/mockup-sandbox: Component Preview Server** — Vite dev server

## Features

### Upload Flow
- Accepts PDF, PNG, JPG, JPEG, TIFF, BMP, DOCX, TXT
- Parses documents using **LlamaParse** (cloud OCR + parsing)
- Chunks text into ~2000 character chunks with 300 char overlap
- Embeds chunks using ChromaDB's default embedding (local sentence-transformers)
- Stores chunks + metadata (filename, chunk_index) in **Chroma** persistent vector DB

### Query Flow
- Embeds user query with ChromaDB's embedding function
- Retrieves top-5 most relevant chunks via cosine similarity
- Constructs a medical/billing assistant prompt with retrieved context
- Calls selected LLM to generate an answer
- Returns answer + source citations (filename + chunk snippets)

### Multi-LLM Support
Users can switch LLM providers and models from the UI dropdown:

| Provider | Models | Free |
|---|---|---|
| **Groq** (default) | Llama 3.1/3.3 70B, Llama 3.1 8B, Mixtral 8x7B, Gemma 2 9B | ✅ |
| **Google Gemini** | Gemini 1.5 Flash/Pro, Gemini 2.0 Flash | ✅ (free tier) |
| **OpenRouter** | Llama 3.1 70B, Mistral 7B, Gemma 2 9B, Phi-3 Mini (all free) | ✅ |
| **OpenAI** | GPT-4o Mini, GPT-4o | ❌ paid |

## Environment Secrets Required

| Secret | Purpose | Where to get |
|---|---|---|
| `GROQ_API_KEY` | LLM inference (default) | console.groq.com (free) |
| `LLAMAPARSE_API_KEY` | Document parsing/OCR | cloud.llamaindex.ai |
| `GEMINI_API_KEY` | Optional: Gemini models | aistudio.google.com |
| `OPENROUTER_API_KEY` | Optional: OpenRouter models | openrouter.ai |
| `OPENAI_API_KEY` | Optional: OpenAI models | platform.openai.com |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/providers` | List LLM providers and their status |
| GET | `/documents` | List indexed documents |
| POST | `/upload` | Upload and index a document |
| POST | `/query` | Query documents using RAG |
| DELETE | `/documents/{filename}` | Remove a document from the index |

## Running Locally

```bash
pip install -r artifacts/rag-app/requirements.txt
cd artifacts/rag-app
uvicorn main:app --reload --port 5000
# Open http://localhost:5000
```

## Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Document Parsing**: LlamaParse (cloud API)
- **Vector DB**: ChromaDB (local persistent, cosine similarity)
- **Embeddings**: ChromaDB default (all-MiniLM-L6-v2 via ONNX, local/free)
- **LLMs**: Groq, Google Gemini, OpenRouter, OpenAI (switchable)
- **Frontend**: Vanilla HTML + JavaScript (no frameworks)
