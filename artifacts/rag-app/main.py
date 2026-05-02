import os
import uuid
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Document QA")

LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "./chroma_db"
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

PROVIDERS = {
    "groq": {
        "name": "Groq (Free)",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B"},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B (Fast)"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B"},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B"},
            {"id": "llama3-70b-8192", "name": "Llama 3 70B"},
        ],
        "env_key": "GROQ_API_KEY",
        "free": True,
    },
    "gemini": {
        "name": "Google Gemini (Free tier)",
        "models": [
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
        ],
        "env_key": "GEMINI_API_KEY",
        "free": True,
    },
    "openrouter": {
        "name": "OpenRouter (Free models)",
        "models": [
            {"id": "meta-llama/llama-3.1-70b-instruct:free", "name": "Llama 3.1 70B"},
            {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B"},
            {"id": "google/gemma-2-9b-it:free", "name": "Gemma 2 9B"},
            {"id": "microsoft/phi-3-mini-128k-instruct:free", "name": "Phi-3 Mini"},
        ],
        "env_key": "OPENROUTER_API_KEY",
        "free": True,
    },
    "openai": {
        "name": "OpenAI",
        "models": [
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
            {"id": "gpt-4o", "name": "GPT-4o"},
        ],
        "env_key": "OPENAI_API_KEY",
        "free": False,
    },
}


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 300) -> list[str]:
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


async def parse_document_llamaparse(file_path: str) -> str:
    if not LLAMAPARSE_API_KEY:
        raise HTTPException(status_code=500, detail="LLAMAPARSE_API_KEY is not configured.")
    from llama_parse import LlamaParse
    parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="text")
    documents = await parser.aload_data(file_path)
    return "\n\n".join([doc.text for doc in documents])


async def call_llm(provider: str, model: str, prompt: str) -> str:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    env_key = PROVIDERS[provider]["env_key"]
    api_key = os.getenv(env_key)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"API key {env_key} is not set. Please add it to your environment secrets.",
        )

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        return response.text

    elif provider == "openrouter":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content

    raise HTTPException(status_code=400, detail="Provider not implemented.")


@app.get("/providers")
def get_providers():
    result = {}
    for key, val in PROVIDERS.items():
        env_key = val["env_key"]
        configured = bool(os.getenv(env_key))
        result[key] = {**val, "configured": configured}
    return result


@app.get("/documents")
def list_documents():
    try:
        results = collection.get(include=["metadatas"])
        filenames = {}
        for meta in results["metadatas"]:
            fn = meta.get("filename", "unknown")
            filenames[fn] = filenames.get(fn, 0) + 1
        return [{"filename": k, "chunks": v} for k, v in filenames.items()]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"Parsing {file.filename} with LlamaParse...")
        text = await parse_document_llamaparse(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

        chunks = chunk_text(text)
        logger.info(f"Split into {len(chunks)} chunks")

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"filename": file.filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info(f"Stored {len(chunks)} chunks for {file.filename}")

        return {
            "success": True,
            "filename": file.filename,
            "chunks_stored": len(chunks),
            "preview": text[:300] + "..." if len(text) > 300 else text,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class QueryRequest(BaseModel):
    query: str
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    top_k: int = 5


@app.post("/query")
async def query_documents(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = collection.query(
            query_texts=[req.query],
            n_results=min(req.top_k, collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Chroma query error: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return {"answer": "No documents found in the database. Please upload some documents first.", "sources": []}

    context = "\n\n---\n\n".join(
        [f"[Source: {m['filename']}, chunk {m['chunk_index']}]\n{d}" for d, m in zip(docs, metas)]
    )

    prompt = f"""You are a medical and billing assistant. Use the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{req.query}

Answer clearly and concisely, and explain medical or billing terms if present."""

    logger.info(f"Calling {req.provider}/{req.model} for query: {req.query[:60]}...")
    answer = await call_llm(req.provider, req.model, prompt)

    sources = [
        {"filename": m["filename"], "chunk_index": m["chunk_index"], "snippet": d[:200] + "..." if len(d) > 200 else d}
        for d, m in zip(docs, metas)
    ]

    return {"answer": answer, "sources": sources}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    try:
        results = collection.get(where={"filename": filename}, include=["metadatas"])
        ids = results["ids"]
        if not ids:
            raise HTTPException(status_code=404, detail="Document not found.")
        collection.delete(ids=ids)
        return {"success": True, "deleted_chunks": len(ids)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")
