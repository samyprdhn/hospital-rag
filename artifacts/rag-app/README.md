# RAG Document QA

A full-stack Retrieval-Augmented Generation (RAG) app. Upload documents, ask questions, and get AI-powered answers with source citations. Supports multiple LLM providers and document parsers.

---

## Features

- Upload PDFs, images, DOCX, and TXT files
- Ask questions and get answers with source references
- Switch between LLM providers: Groq, Gemini, OpenRouter, OpenAI
- Choose document parsers: LlamaParse, PyMuPDF, pdfplumber, Tesseract OCR
- Compare two parsers side-by-side and index the winner directly
- Persistent vector store via ChromaDB

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- At minimum, a **Groq API key** (free at [console.groq.com](https://console.groq.com))
- Optional: LlamaParse, Gemini, OpenRouter, or OpenAI API keys

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/samyprdhn/hospital-rag.git
cd hospital-rag/artifacts/rag-app
```

### 2. Build the Docker image

```bash
docker build -t hospital-rag .
```

### 3. Run the container

```bash
docker run -d \
  --name hospital-rag \
  -p 8080:8080 \
  -e GROQ_API_KEY=your_groq_api_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  hospital-rag
```

> The `-v` flag mounts a local folder so your indexed documents persist across container restarts.

### 4. Open the app

```
http://localhost:8080
```

---

## With Optional API Keys

Pass any extra keys you have as additional `-e` flags:

```bash
docker run -d \
  --name hospital-rag \
  -p 8080:8080 \
  -e GROQ_API_KEY=your_groq_key \
  -e LLAMAPARSE_API_KEY=your_llamaparse_key \
  -e GEMINI_API_KEY=your_gemini_key \
  -e OPENROUTER_API_KEY=your_openrouter_key \
  -e OPENAI_API_KEY=your_openai_key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  hospital-rag
```

---

## Using an `.env` File

Create a `.env` file in the `artifacts/rag-app` folder:

```env
GROQ_API_KEY=your_groq_key
LLAMAPARSE_API_KEY=your_llamaparse_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key
```

Then run with:

```bash
docker run -d \
  --name hospital-rag \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  hospital-rag
```

---

## Useful Docker Commands

| Action | Command |
|---|---|
| View live logs | `docker logs -f hospital-rag` |
| Stop the container | `docker stop hospital-rag` |
| Start it again | `docker start hospital-rag` |
| Remove the container | `docker rm hospital-rag` |
| Rebuild after code changes | `docker build -t hospital-rag . && docker stop hospital-rag && docker rm hospital-rag` |

Then re-run the `docker run` command above.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `POST` | `/upload` | Upload and index a document |
| `POST` | `/query` | Ask a question |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{filename}` | Delete a document |
| `GET` | `/providers` | List available LLM providers |
| `GET` | `/parsers` | List available parsers |
| `POST` | `/compare` | Compare two parsers on the same file |
| `POST` | `/index-text` | Index already-extracted text directly |

---

## Data Persistence

ChromaDB stores vectors in the `chroma_db/` folder inside the container. The `-v $(pwd)/chroma_db:/app/chroma_db` volume flag maps this to your local machine so data survives container restarts.

If you skip the `-v` flag, all indexed documents are lost when the container stops.

---

## Supported File Types

| Parser | Supported Formats |
|---|---|
| LlamaParse (cloud) | PDF, DOCX, PNG, JPG, TIFF |
| PyMuPDF (local) | PDF, PNG, JPG, TIFF, BMP |
| pdfplumber (local) | PDF |
| Tesseract OCR (local) | PNG, JPG, TIFF, BMP, PDF |
