"""
RAG (Retrieval-Augmented Generation) module for RadiantCare.

At startup, loads .txt/.pdf docs from pathologies/, chunks them, embeds
each chunk via OpenAI, and stores everything in memory.  At query time,
embeds the query, computes cosine similarity against all chunks, and
returns the top-K most relevant ones to inject into the LLM prompt.

Search is purely semantic (embedding similarity), not filename-based,
so label/filename mismatches (e.g. "Support Devices" vs "Supporting_Devices.pdf")
don't matter — the content determines the match.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PATHOLOGIES_DIR_ENV = "PATHOLOGIES_DIR"

DEFAULT_PATHOLOGIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pathologies",
)

CHUNK_SIZE = 600       # chars per chunk — enough context without losing focus
CHUNK_OVERLAP = 100    # overlap to avoid cutting sentences at boundaries
TOP_K = 5              # chunks returned per query
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536-dim, fast and cheap

# ---------------------------------------------------------------------------
# Module-level state (populated once by build_rag_index at startup)
# ---------------------------------------------------------------------------

_embeddings: np.ndarray | None = None   # (num_chunks, 1536)
_chunks: List[str] = []
_num_documents: int = 0
_rag_ready: bool = False
_rag_error: str | None = None


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _get_pathologies_dir() -> str:
    return os.getenv(PATHOLOGIES_DIR_ENV, DEFAULT_PATHOLOGIES_DIR)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    """Extract text from a PDF via PyMuPDF.  Returns '' if not installed."""
    try:
        import fitz
    except ImportError:
        return ""

    text_parts = []
    doc = fitz.open(path)
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def _chunk_text(text: str, source: str) -> List[str]:
    """
    Split text into overlapping chunks prefixed with [source].

    Sliding window: each chunk starts (CHUNK_SIZE - CHUNK_OVERLAP) chars
    after the previous one, so consecutive chunks share CHUNK_OVERLAP chars.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(f"[{source}]\n{chunk.strip()}")
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    return chunks


def _load_documents() -> List[Tuple[str, str]]:
    """Read every .txt/.pdf from pathologies/ → list of (text, source_name)."""
    base = _get_pathologies_dir()
    if not os.path.isdir(base):
        return []

    out = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            continue

        name_no_ext = os.path.splitext(name)[0]

        if name.lower().endswith(".txt"):
            try:
                text = _read_txt(path)
                if text.strip():
                    out.append((text, name_no_ext))
            except Exception:
                pass

        elif name.lower().endswith(".pdf"):
            try:
                text = _read_pdf(path)
                if text.strip():
                    out.append((text, name_no_ext))
            except Exception:
                pass

    return out


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _get_embeddings_openai(texts: List[str]) -> np.ndarray:
    """Embed texts via OpenAI in batches of 50.  Returns (len(texts), 1536)."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — cannot embed documents for RAG")

    client = OpenAI(api_key=api_key)

    batch_size = 50
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        for item in resp.data:
            all_emb.append(item.embedding)

    return np.array(all_emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Build and query the index
# ---------------------------------------------------------------------------

def build_rag_index() -> None:
    """Load docs → chunk → embed → store in memory.  Called once at startup."""
    global _embeddings, _chunks, _num_documents, _rag_ready, _rag_error

    _rag_ready = False
    _rag_error = None
    _chunks = []
    _num_documents = 0

    docs = _load_documents()
    if not docs:
        _rag_error = "No documents found in pathologies directory"
        return
    _num_documents = len(docs)

    for text, source in docs:
        _chunks.extend(_chunk_text(text, source))
    if not _chunks:
        _rag_error = "No chunks extracted from documents"
        return

    try:
        _embeddings = _get_embeddings_openai(_chunks)
        _rag_ready = True
    except Exception as e:
        _rag_error = str(e)
        _embeddings = None


def get_rag_context(query: str, k: int = TOP_K) -> str:
    """
    Embed the query, cosine-similarity search against all chunks, return
    the top-k as a single string separated by '---'.

    Returns '' if RAG is not ready or query is empty.
    """
    global _embeddings, _chunks, _rag_ready
    if not query or not _rag_ready or _embeddings is None or not _chunks:
        return ""

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception:
        return ""

    # Normalize chunk embeddings and query → cosine sim = dot product
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = _embeddings / norms

    q_norm = q_emb / (np.linalg.norm(q_emb) or 1.0)

    sims = np.dot(emb_norm, q_norm)

    top_idx = np.argsort(sims)[::-1][:k]

    parts = [_chunks[i] for i in top_idx]
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Status helpers (used by /health)
# ---------------------------------------------------------------------------

def is_rag_ready() -> bool:
    return _rag_ready


def get_rag_error() -> str | None:
    return _rag_error


def get_num_documents() -> int:
    return _num_documents


def get_num_chunks() -> int:
    return len(_chunks)
