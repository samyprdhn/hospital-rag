import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import uuid
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
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

# ── Parser registry ────────────────────────────────────────────────────────────
PARSERS = {
    "llamaparse": {
        "name": "LlamaParse",
        "description": "Cloud API — best quality, handles complex layouts, tables & images",
        "badge": "Cloud",
        "env_key": "LLAMAPARSE_API_KEY",
        "free": False,
        "formats": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx"],
    },
    "pymupdf": {
        "name": "PyMuPDF",
        "description": "Local — fast native PDF text extraction, no API key needed",
        "badge": "Local · Free",
        "env_key": None,
        "free": True,
        "formats": [".pdf"],
    },
    "pdfplumber": {
        "name": "pdfplumber",
        "description": "Local — excellent table extraction from PDFs, no API key needed",
        "badge": "Local · Free",
        "env_key": None,
        "free": True,
        "formats": [".pdf"],
    },
    "tesseract": {
        "name": "Tesseract OCR",
        "description": "Local — OCR for images and scanned PDFs (converted page by page)",
        "badge": "Local · Free",
        "env_key": None,
        "free": True,
        "formats": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
    },
}

# ── LLM provider registry ──────────────────────────────────────────────────────
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


# ── Parsing functions ──────────────────────────────────────────────────────────

async def parse_with_llamaparse(file_path: str) -> str:
    if not LLAMAPARSE_API_KEY:
        raise HTTPException(status_code=400, detail="LLAMAPARSE_API_KEY is not configured.")
    from llama_parse import LlamaParse
    parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="text")
    documents = await parser.aload_data(file_path)
    return "\n\n".join([doc.text for doc in documents])


def parse_with_pymupdf(file_path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def parse_with_pdfplumber(file_path: str) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    if row:
                        table_text += " | ".join(str(c or "") for c in row) + "\n"
            pages.append((text + "\n" + table_text).strip())
    return "\n\n".join(p for p in pages if p)


def parse_with_tesseract(file_path: str, suffix: str) -> str:
    import pytesseract
    from PIL import Image

    if suffix == ".pdf":
        import fitz  # PyMuPDF to render pages
        doc = fitz.open(file_path)
        texts = []
        for page in doc:
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 dpi
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            import io
            img = Image.open(io.BytesIO(img_data))
            texts.append(pytesseract.image_to_string(img))
        doc.close()
        return "\n\n".join(texts)
    else:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)


async def parse_document(file_path: str, suffix: str, parser: str) -> str:
    if parser not in PARSERS:
        raise HTTPException(status_code=400, detail=f"Unknown parser: {parser}")

    p = PARSERS[parser]
    if suffix not in p["formats"]:
        supported = ", ".join(p["formats"])
        raise HTTPException(
            status_code=400,
            detail=f"Parser '{p['name']}' does not support {suffix} files. Supported: {supported}",
        )

    if p["env_key"] and not os.getenv(p["env_key"]):
        raise HTTPException(
            status_code=400,
            detail=f"API key {p['env_key']} is required for {p['name']} but is not configured.",
        )

    logger.info(f"Parsing with {p['name']} ...")

    if parser == "llamaparse":
        return await parse_with_llamaparse(file_path)
    elif parser == "pymupdf":
        return parse_with_pymupdf(file_path)
    elif parser == "pdfplumber":
        return parse_with_pdfplumber(file_path)
    elif parser == "tesseract":
        return parse_with_tesseract(file_path, suffix)

    raise HTTPException(status_code=400, detail="Parser not implemented.")


# ── LLM call ───────────────────────────────────────────────────────────────────

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
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
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


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/parsers")
def get_parsers():
    result = {}
    for key, val in PARSERS.items():
        env_key = val.get("env_key")
        configured = True if not env_key else bool(os.getenv(env_key))
        result[key] = {**val, "configured": configured}
    return result


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
        docs: dict[str, dict] = {}
        for meta in results["metadatas"]:
            fn = meta.get("filename", "unknown")
            if fn not in docs:
                docs[fn] = {"filename": fn, "chunks": 0, "parser": meta.get("parser", "unknown")}
            docs[fn]["chunks"] += 1
        return list(docs.values())
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    parser: str = Form("llamaparse"),
):
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        text = await parse_document(tmp_path, suffix, parser)

        if not text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

        chunks = chunk_text(text)
        logger.info(f"Split into {len(chunks)} chunks")

        ids = [str(uuid.uuid4()) for _ in chunks]
        parser_name = PARSERS[parser]["name"]
        metadatas = [
            {"filename": file.filename, "chunk_index": i, "parser": parser_name}
            for i in range(len(chunks))
        ]

        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info(f"Stored {len(chunks)} chunks for {file.filename} via {parser_name}")

        return {
            "success": True,
            "filename": file.filename,
            "parser": parser_name,
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

Currency rules (strictly follow these):
- Always use the exact currency symbol and amount as it appears in the source document.
- Do not convert, substitute, or guess the currency (e.g. do not replace R with $ or £).
- If the document contains clues about its country of origin (e.g. country name, city, regulatory body, bank, insurer, phone format, address), infer the correct local currency from that country and use it consistently. Examples: South Africa → R (Rand), Nepal → Rs / रू (Nepali Rupee), India → ₹ (Rupee), UK → £ (Pound), USA → $ (Dollar), Europe → € (Euro), Kenya → KSh (Shilling).
- If the currency symbol is genuinely absent and the country cannot be determined from the document, state "currency not specified in the document" rather than guessing.

Context:
{context}

Question:
{req.query}

Answer clearly and concisely, and explain medical or billing terms if present."""

    logger.info(f"Calling {req.provider}/{req.model} for query: {req.query[:60]}...")
    answer = await call_llm(req.provider, req.model, prompt)

    sources = [
        {
            "filename": m["filename"],
            "chunk_index": m["chunk_index"],
            "parser": m.get("parser", "unknown"),
            "snippet": d[:200] + "..." if len(d) > 200 else d,
        }
        for d, m in zip(docs, metas)
    ]

    return {"answer": answer, "sources": sources}


class IndexTextRequest(BaseModel):
    filename: str
    parser_key: str
    text: str


@app.post("/index-text")
async def index_text(req: IndexTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty — nothing to index.")
    if req.parser_key not in PARSERS:
        raise HTTPException(status_code=400, detail=f"Unknown parser: {req.parser_key}")

    parser_name = PARSERS[req.parser_key]["name"]
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No usable chunks produced from the text.")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"filename": req.filename, "chunk_index": i, "parser": parser_name}
        for i in range(len(chunks))
    ]
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    logger.info(f"Indexed {len(chunks)} chunks for '{req.filename}' via {parser_name} (from compare)")

    return {
        "success": True,
        "filename": req.filename,
        "parser": parser_name,
        "chunks_stored": len(chunks),
    }


@app.post("/compare")
async def compare_parsers(
    file: UploadFile = File(...),
    parser_a: str = Form(...),
    parser_b: str = Form(...),
):
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    if parser_a not in PARSERS or parser_b not in PARSERS:
        raise HTTPException(status_code=400, detail="Unknown parser specified.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        import time
        results = {}
        for key in (parser_a, parser_b):
            if key not in results:
                p = PARSERS[key]
                if suffix not in p["formats"]:
                    results[key] = {
                        "parser": p["name"],
                        "error": f"Parser does not support {suffix} files.",
                        "text": "",
                        "chars": 0,
                        "words": 0,
                        "lines": 0,
                        "chunks": 0,
                        "elapsed_ms": 0,
                    }
                    continue
                t0 = time.time()
                try:
                    text = await parse_document(tmp_path, suffix, key)
                    elapsed = int((time.time() - t0) * 1000)
                    chunks = chunk_text(text)
                    results[key] = {
                        "parser": p["name"],
                        "text": text,
                        "preview": text[:4000],
                        "chars": len(text),
                        "words": len(text.split()),
                        "lines": text.count("\n") + 1,
                        "chunks": len(chunks),
                        "elapsed_ms": elapsed,
                        "error": None,
                    }
                except HTTPException as e:
                    results[key] = {
                        "parser": p["name"],
                        "error": e.detail,
                        "text": "",
                        "preview": "",
                        "chars": 0,
                        "words": 0,
                        "lines": 0,
                        "chunks": 0,
                        "elapsed_ms": 0,
                    }

        return {
            "filename": file.filename,
            "a": results[parser_a],
            "b": results[parser_b],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
