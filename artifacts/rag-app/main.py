import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import uuid
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
import sqlite3
import secrets
import hashlib

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

# ── User Authentication Setup ──────────────────────────────────────────────────
DB_PATH = Path(CHROMA_PATH) / "users.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        conn.commit()

init_db()

def hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    if not salt_hex:
        salt_hex = secrets.token_hex(16)
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return key.hex(), salt_hex

class UserSignUp(BaseModel):
    username: str
    email: str
    password: str

class UserSignIn(BaseModel):
    username: str
    password: str

class UserInfo(BaseModel):
    id: str
    username: str
    email: str

class AuthResponse(BaseModel):
    token: str
    user: UserInfo

@app.post("/auth/signup", response_model=AuthResponse)
def signup(req: UserSignUp):
    if not req.username.strip() or not req.email.strip() or not req.password:
        raise HTTPException(status_code=400, detail="All fields (username, email, password) are required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    user_id = str(uuid.uuid4())
    pw_hash, salt = hash_password(req.password)
    token = secrets.token_hex(32)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, salt) VALUES (?, ?, ?, ?, ?)",
                (user_id, req.username.strip().lower(), req.email.strip().lower(), pw_hash, salt)
            )
            conn.execute(
                "INSERT INTO sessions (token, user_id) VALUES (?, ?)",
                (token, user_id)
            )
            conn.commit()
    except sqlite3.IntegrityError as e:
        if "username" in str(e).lower() or "users.username" in str(e).lower():
            raise HTTPException(status_code=400, detail="Username is already taken.")
        if "email" in str(e).lower() or "users.email" in str(e).lower():
            raise HTTPException(status_code=400, detail="Email is already registered.")
        raise HTTPException(status_code=400, detail="Username or email already exists.")

    return AuthResponse(
        token=token,
        user=UserInfo(id=user_id, username=req.username.strip(), email=req.email.strip().lower())
    )

@app.post("/auth/signin", response_model=AuthResponse)
def signin(req: UserSignIn):
    login_id = req.username.strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT id, username, email, password_hash, salt FROM users WHERE username = ? OR email = ?",
            (login_id, login_id)
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    user_id, username, email, pw_hash, salt = row
    check_hash, _ = hash_password(req.password, salt)
    if check_hash != pw_hash:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = secrets.token_hex(32)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))
        conn.commit()

    return AuthResponse(
        token=token,
        user=UserInfo(id=user_id, username=username, email=email)
    )

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in or create an account.")
    token = authorization.split("Bearer ", 1)[1].strip()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """SELECT u.id, u.username, u.email FROM sessions s
               JOIN users u ON s.user_id = u.id
               WHERE s.token = ?""",
            (token,)
        )
        row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token. Please sign in again.")
    return {"id": row[0], "username": row[1], "email": row[2]}

@app.get("/auth/me", response_model=UserInfo)
def get_me(current_user: dict = Depends(get_current_user)):
    return UserInfo(**current_user)

@app.post("/auth/signout")
def signout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
    return {"success": True}

def get_user_collection(user_id: str):
    safe_name = f"user_{user_id.replace('-', '_')}"
    return chroma_client.get_or_create_collection(
        name=safe_name,
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
    
    logger.info("🔍 Running dual-pass LlamaParse: English + Nepali extraction...")
    
    # ── First pass: English extraction ──
    logger.info("📄 Pass 1/2: Extracting English text...")
    parser_en = LlamaParse(
        api_key=LLAMAPARSE_API_KEY, 
        result_type="text",
        language="en",
        verbose=False
    )
    documents_en = await parser_en.aload_data(file_path)
    text_en = "\n\n".join([doc.text for doc in documents_en])
    logger.info(f"   ✓ English pass complete ({len(text_en)} characters)")
    
    # ── Second pass: Nepali extraction ──
    logger.info("🇳🇵 Pass 2/2: Extracting Nepali text...")
    parser_ne = LlamaParse(
        api_key=LLAMAPARSE_API_KEY, 
        result_type="text",
        language="ne",
        verbose=False
    )
    documents_ne = await parser_ne.aload_data(file_path)
    text_ne = "\n\n".join([doc.text for doc in documents_ne])
    logger.info(f"   ✓ Nepali pass complete ({len(text_ne)} characters)")
    
    # ── Merge results ──
    logger.info("🔀 Merging extraction results...")
    merged_text = _merge_parsed_texts(text_en, text_ne)
    logger.info(f"   ✓ Merge complete ({len(merged_text)} characters)")
    
    return merged_text


def _merge_parsed_texts(text_en: str, text_ne: str) -> str:
    """
    Intelligently merge English and Nepali extracted text.
    
    Strategy:
    1. If text_ne is empty or too short, use English only (no Nepali content found)
    2. If text_en is empty but text_ne exists, use Nepali only
    3. Otherwise, interleave them intelligently for better context preservation
    
    Returns: Merged text optimized for RAG vector embedding
    """
    
    # Remove extra whitespace
    text_en = text_en.strip()
    text_ne = text_ne.strip()
    
    # If no Nepali text was found, return English only
    if not text_ne or len(text_ne) < 100:
        logger.info(f"   → No significant Nepali text found. Using English extraction only.")
        return text_en
    
    # If no English text but Nepali exists, return Nepali only
    if not text_en or len(text_en) < 100:
        logger.info(f"   → No significant English text found. Using Nepali extraction only.")
        return text_ne
    
    # Both languages present - create comprehensive merged version
    logger.info(f"   → Bilingual content detected. Merging {len(text_en)} chars (EN) + {len(text_ne)} chars (NE)")
    
    # Split into paragraphs
    paragraphs_en = [p.strip() for p in text_en.split('\n\n') if p.strip()]
    paragraphs_ne = [p.strip() for p in text_ne.split('\n\n') if p.strip()]
    
    merged = []
    
    # Strategy: If roughly same number of paragraphs, they're likely aligned translations
    # Interleave them for maximum context preservation in RAG
    if abs(len(paragraphs_en) - len(paragraphs_ne)) <= 2:
        logger.info(f"   → Paragraph count similar ({len(paragraphs_en)} vs {len(paragraphs_ne)}), interleaving...")
        # Interleave English and Nepali paragraphs
        for i in range(max(len(paragraphs_en), len(paragraphs_ne))):
            if i < len(paragraphs_en):
                merged.append(paragraphs_en[i])
            if i < len(paragraphs_ne):
                # Add metadata marker for Nepali content
                merged.append(f"[नेपाली / Nepali]\n{paragraphs_ne[i]}")
    else:
        # Different number of paragraphs - section them separately with markers
        logger.info(f"   → Paragraph count different ({len(paragraphs_en)} vs {len(paragraphs_ne)}), sectioning...")
        merged.append("=" * 50)
        merged.append("ENGLISH EXTRACTION (अंग्रेजी पाठ)")
        merged.append("=" * 50)
        merged.extend(paragraphs_en)
        merged.append("")
        merged.append("=" * 50)
        merged.append("NEPALI EXTRACTION (नेपाली पाठ)")
        merged.append("=" * 50)
        merged.extend(paragraphs_ne)
    
    merged_result = "\n\n".join(merged)
    logger.info(f"   ✓ Final merged text: {len(merged_result)} characters")
    return merged_result


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
            # Support both English and Nepali (Devanagari script)
            texts.append(pytesseract.image_to_string(img, lang="eng+nep"))
        doc.close()
        return "\n\n".join(texts)
    else:
        img = Image.open(file_path)
        # Support both English and Nepali (Devanagari script)
        return pytesseract.image_to_string(img, lang="eng+nep")


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
def list_documents(current_user: dict = Depends(get_current_user)):
    try:
        user_col = get_user_collection(current_user["id"])
        results = user_col.get(include=["metadatas"])
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
    current_user: dict = Depends(get_current_user),
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

        user_col = get_user_collection(current_user["id"])
        user_col.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info(f"Stored {len(chunks)} chunks for {file.filename} via {parser_name} for user {current_user['username']}")

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
async def query_documents(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        user_col = get_user_collection(current_user["id"])
        results = user_col.query(
            query_texts=[req.query],
            n_results=min(req.top_k, user_col.count() or 1),
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

    prompt = f"""You are a medical document and billing assistant for a hospital system. Answer using ONLY the context below — never use outside medical knowledge to fill gaps.

    Scope and safety:
    1. You are retrieving and explaining information that already exists in the patient's documents — you are NOT diagnosing, recommending treatment, or giving clinical advice. If asked something that requires clinical judgment beyond what's written (e.g. "should I take this medication," "is this dosage safe"), answer only with what the document states and add: "This is informational only — please confirm with a clinician."
    2. Never infer a diagnosis, condition, or billing code that isn't explicitly stated in the context, even if symptoms or line items seem to imply one.
    3. If the answer is not in the context, say "I don't know based on the provided document(s)." Do not guess.
    
    Accuracy requirements:
    4. Reproduce all codes (ICD-10, CPT, HCPCS, NDC), dates, amounts, claim numbers, policy numbers, and provider/patient identifiers EXACTLY as written. Never round, reformat, recalculate, or auto-correct what looks like a typo.
    5. When explaining a code, state the code as written, then its plain-language meaning if commonly known (e.g. "CPT 99213 — established patient office visit, low-to-moderate complexity"). If you're not certain what a specific code means, say so rather than guessing.
    6. Briefly explain insurance/billing terms in plain language the first time they appear (e.g. deductible, co-pay, co-insurance, EOB, prior authorization, out-of-pocket max, allowed amount, adjustment).
    7. If the question is about a resume, letter, or other non-medical document, just answer normally from the context — don't force a medical framing.
    
    Deduplication (important):
    8. The context below may contain the same line item, fact, or statement repeated across multiple retrieved passages (e.g. the same charge appearing more than once due to how the document was split for search). Treat repeated occurrences of the same description, date, and amount as ONE fact. Do not list duplicates as if they were separate charges or separate events, and do not comment on the fact that retrieval returned duplicates.
    9. Only flag something as a genuine discrepancy if two passages describe what looks like the same item/date but with DIFFERING amounts, codes, or details. In that case, show both versions and say they conflict.
    
    Handling multiple documents/encounters:
    10. Hospital records often span multiple visits, claims, or providers. When it's useful for clarity — e.g. the question involves more than one date, claim, or document — attribute facts using natural references like the document name, visit date, or claim number (e.g. "per the 03/12/2026 visit note," "on claim #12345"). Do not reference internal retrieval or system details such as chunk numbers, chunk indices, or passage numbers — these are not meaningful to the reader.
    11. For simple questions with one clear answer, just give the answer plainly — don't manufacture a source citation for every sentence if the whole answer comes from one obvious place.
    12. If the source document contains a stated summary figure (e.g. "Total Due," "Balance," "Amount Payable"), use that stated figure directly rather than adding up individual line items yourself.
    
    Currency rules (strict):
    - Use the exact currency symbol and amount as it appears in the source. Never convert or substitute (e.g. never replace R with $ or £).
    - If no symbol is present, infer the currency from country/region clues in the document (country name, city, regulatory body, bank, insurer, phone format, address). Examples: South Africa → R, Nepal → Rs / रू, India → ₹, UK → £, USA → $, Eurozone → €, Kenya → KSh.
    - If there's no symbol AND no locatable country clue, say "currency not specified in the document" instead of guessing.
    
    Privacy:
    13. Only surface patient-identifying details (name, DOB, SSN, MRN, policy number) if they are directly relevant to answering the question — don't restate them unnecessarily.
    
    Context:
    {context}
    
    Question:
    {req.query}
    
    Answer clearly and concisely in plain, natural language. Attribute facts to their source (document/date/claim) only where it adds clarity — never mention chunks, passages, or retrieval mechanics."""

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
async def index_text(req: IndexTextRequest, current_user: dict = Depends(get_current_user)):
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
    user_col = get_user_collection(current_user["id"])
    user_col.add(documents=chunks, metadatas=metadatas, ids=ids)
    logger.info(f"Indexed {len(chunks)} chunks for '{req.filename}' via {parser_name} for user {current_user['username']} (from compare)")

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
    current_user: dict = Depends(get_current_user),
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
def delete_document(filename: str, current_user: dict = Depends(get_current_user)):
    try:
        user_col = get_user_collection(current_user["id"])
        results = user_col.get(where={"filename": filename}, include=["metadatas"])
        ids = results["ids"]
        if not ids:
            raise HTTPException(status_code=404, detail="Document not found.")
        user_col.delete(ids=ids)
        return {"success": True, "deleted_chunks": len(ids)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def serve_index():
    from fastapi.responses import HTMLResponse
    with open("static/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")
