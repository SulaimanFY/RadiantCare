"""
RAG (Retrieval-Augmented Generation) module for RadiantCare.

What is RAG?
  A technique where, instead of relying solely on the LLM's training data, we
  retrieve relevant text passages from a local knowledge base and inject them
  into the LLM prompt. This gives the LLM access to specific, up-to-date
  information it wouldn't otherwise have (e.g. pathology treatment protocols).

How it works in this project:
  1. At startup, load all .txt and .pdf documents from the pathologies/ folder.
     Each file describes one chest pathology (symptoms, diagnosis, treatment).
  2. Split each document into overlapping text chunks (~600 characters each).
     Overlap ensures we don't lose context at chunk boundaries.
  3. Embed each chunk using OpenAI's text-embedding-3-small model.
     "Embedding" = converting text into a numerical vector that captures its
     meaning; semantically similar texts produce similar vectors.
  4. Store all chunks and their embedding vectors in memory (numpy arrays).
  5. When the user uploads an X-ray (or asks a question), we build a query
     string from the predictions + clinical context, embed it with the same
     model, compute cosine similarity against all stored chunk embeddings,
     and return the top-K most relevant chunks.
  6. These chunks are injected into the LLM prompt as "Relevant excerpts
     from protocol/pathology documents", so the LLM can reference real
     medical protocol data in its answer.

Note on search vs. file names:
  The RAG does NOT filter by file name or label name. It relies entirely on
  semantic similarity between the query and the chunk content. So even if a
  file is named "Supporting_Devices.pdf" while the model label says "Support
  Devices", the search still works because the file's *content* (not its name)
  determines the match. The file name only appears as a prefix in each chunk
  (e.g. "[Supporting_Devices]") so the LLM knows the source.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Environment variable to override the default pathologies folder path
PATHOLOGIES_DIR_ENV = "PATHOLOGIES_DIR"

# Default: <project_root>/pathologies/
DEFAULT_PATHOLOGIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # project root
    "pathologies",
)

# Chunking parameters:
# - CHUNK_SIZE: max characters per chunk. 600 is a good balance between
#   having enough context and keeping embeddings focused.
# - CHUNK_OVERLAP: characters shared between consecutive chunks.
#   100 chars of overlap prevents cutting a sentence in half and losing meaning.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# How many chunks to return per query. 5 gives the LLM enough context
# without flooding the prompt (which costs tokens and can confuse the LLM).
TOP_K = 5

# OpenAI embedding model. "text-embedding-3-small" is fast, cheap, and produces
# 1536-dimensional vectors. We use the same model for both indexing and querying
# so the vectors live in the same space and are comparable.
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# These variables are populated once by build_rag_index() at startup.
# They are module-level (not inside a class) for simplicity.
# In a larger app you might use a singleton or dependency injection.

_embeddings: np.ndarray | None = None   # shape (num_chunks, 1536), float32
_chunks: List[str] = []                 # the text content of each chunk
_num_documents: int = 0                 # how many files were loaded (for /health)
_rag_ready: bool = False                # True if index was built successfully
_rag_error: str | None = None           # error message if build failed


# ---------------------------------------------------------------------------
# Document loading helpers
# ---------------------------------------------------------------------------

def _get_pathologies_dir() -> str:
    """Return the path to the pathologies folder (from .env or default)."""
    return os.getenv(PATHOLOGIES_DIR_ENV, DEFAULT_PATHOLOGIES_DIR)


def _read_txt(path: str) -> str:
    """
    Read a plain .txt file and return its contents.
    errors="replace" handles any encoding issues gracefully (replaces bad chars).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    """
    Read a PDF file and extract all text from every page.

    Uses PyMuPDF (pip package "pymupdf", imported as "fitz").
    Returns "" if PyMuPDF is not installed (so the app still works
    for .txt-only setups, just without PDF support).
    """
    try:
        import fitz  # PyMuPDF — pip install pymupdf, but the import name is "fitz"
    except ImportError:
        return ""

    text_parts = []
    doc = fitz.open(path)            # open the PDF file
    for page in doc:                 # iterate over every page in the PDF
        text_parts.append(page.get_text())  # extract the text content of this page
    doc.close()
    return "\n".join(text_parts)     # join all pages into one big string


def _chunk_text(text: str, source: str) -> List[str]:
    """
    Split a document's text into overlapping chunks of ~CHUNK_SIZE characters.

    Why overlapping?
      If we split exactly at character 600, we might cut a sentence in half.
      With overlap, the end of chunk N reappears at the start of chunk N+1,
      so no sentence is completely lost.

    Each chunk is prefixed with [source_name] (e.g. "[Pneumonia]") so the LLM
    knows which document the excerpt comes from when it reads the prompt.

    Example with CHUNK_SIZE=600, CHUNK_OVERLAP=100:
      Chunk 1: characters   0 – 599
      Chunk 2: characters 500 – 1099  (overlaps with chunk 1 by 100 chars)
      Chunk 3: characters 1000 – 1599
      ...
    """
    # Collapse all whitespace (newlines, tabs, multiple spaces) into single spaces
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE              # take up to CHUNK_SIZE characters
        chunk = text[start:end]
        if chunk.strip():
            # Prefix each chunk with [source] so the LLM can cite its origin
            chunks.append(f"[{source}]\n{chunk.strip()}")
        start = end - CHUNK_OVERLAP           # slide the window forward (with overlap)
        if start >= len(text):
            break
    return chunks


def _load_documents() -> List[Tuple[str, str]]:
    """
    Scan the pathologies/ folder and read every .txt and .pdf file.

    Returns a list of (full_text, source_name) tuples.
    source_name = filename without extension, e.g. "Pneumonia" from "Pneumonia.pdf".
    Files are sorted alphabetically for reproducibility.
    """
    base = _get_pathologies_dir()
    if not os.path.isdir(base):
        return []

    out = []
    for name in sorted(os.listdir(base)):    # sorted for deterministic order
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            continue

        # "Pneumonia.pdf" → "Pneumonia" (used as the [source] prefix in chunks)
        name_no_ext = os.path.splitext(name)[0]

        if name.lower().endswith(".txt"):
            try:
                text = _read_txt(path)
                if text.strip():
                    out.append((text, name_no_ext))
            except Exception:
                pass  # skip files that can't be read (corrupted, permissions, etc.)

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
    """
    Embed a list of text strings using OpenAI's embedding API.

    What is an embedding?
      A fixed-size vector (1536 floats for text-embedding-3-small) that captures
      the "meaning" of a text. Texts about similar topics produce vectors that
      point in similar directions in 1536-dimensional space.

    We process texts in batches of 50 to stay within API limits.
    Returns a numpy array of shape (len(texts), 1536).
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — cannot embed documents for RAG")

    client = OpenAI(api_key=api_key)

    batch_size = 50  # OpenAI supports up to ~2048 inputs, but 50 keeps requests small
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Call the embeddings endpoint — returns one embedding per input text
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        # resp.data is a list of EmbeddingObject, each with .embedding (list of floats)
        for item in resp.data:
            all_emb.append(item.embedding)

    # Convert to a numpy array for fast cosine similarity computation later
    return np.array(all_emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Build and query the index
# ---------------------------------------------------------------------------

def build_rag_index() -> None:
    """
    Load all documents, chunk them, embed them, and store in memory.
    Called once at API startup (in main.py, after load_dotenv()).

    After this function runs:
      - _chunks: list of all text chunks (each prefixed with [source])
      - _embeddings: numpy array of shape (num_chunks, 1536)
      - _num_documents: how many source files were loaded
      - _rag_ready: True if everything succeeded
      - _rag_error: error message if something failed (None if OK)
    """
    global _embeddings, _chunks, _num_documents, _rag_ready, _rag_error

    # Reset state (in case this is called again, e.g. during testing)
    _rag_ready = False
    _rag_error = None
    _chunks = []
    _num_documents = 0

    # Step 1: load all .txt and .pdf files from pathologies/
    docs = _load_documents()
    if not docs:
        _rag_error = "No documents found in pathologies directory"
        return
    _num_documents = len(docs)

    # Step 2: split every document into overlapping chunks
    for text, source in docs:
        _chunks.extend(_chunk_text(text, source))
    if not _chunks:
        _rag_error = "No chunks extracted from documents"
        return

    # Step 3: embed all chunks via OpenAI
    # This is the most expensive step (API calls); it runs once at startup.
    try:
        _embeddings = _get_embeddings_openai(_chunks)
        _rag_ready = True
    except Exception as e:
        _rag_error = str(e)
        _embeddings = None


def get_rag_context(query: str, k: int = TOP_K) -> str:
    """
    Find the top-k most relevant chunks for a query and return them as a string.

    Steps:
      1. Embed the query using the same OpenAI model as the indexed chunks.
      2. Compute cosine similarity between the query vector and every chunk vector.
      3. Pick the k chunks with the highest similarity scores.
      4. Concatenate them with "---" separators and return.

    Why cosine similarity?
      cosine_sim(a, b) = dot(a, b) / (|a| × |b|)
      It measures the angle between two vectors, ignoring magnitude.
      Values range from -1 (opposite meaning) to 1 (identical meaning).
      By normalizing vectors to unit length first, cosine similarity = dot product,
      which is very fast to compute with numpy.

    Why does this return the right chunks when the model predicts "Fracture"?
      The query includes the predictions (e.g. "Fracture: positive (probability=0.87)").
      The embedding of this query will be close to chunks that discuss fractures.
      Chunks about cardiomegaly will have lower similarity and won't be selected.
      This is purely semantic — no explicit label-to-file mapping is needed.

    Returns "" if RAG is not ready, or the query is empty.
    """
    global _embeddings, _chunks, _rag_ready
    if not query or not _rag_ready or _embeddings is None or not _chunks:
        return ""

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    # Step 1: embed the query (same model as the chunks)
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32)  # shape: (1536,)
    except Exception:
        return ""

    # Step 2: compute cosine similarity with all chunk embeddings
    # First, normalize all chunk embeddings to unit length
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)  # shape: (num_chunks, 1)
    norms[norms == 0] = 1.0                   # avoid division by zero for empty embeddings
    emb_norm = _embeddings / norms            # each row is now a unit vector

    # Normalize the query embedding too
    q_norm = q_emb / (np.linalg.norm(q_emb) or 1.0)  # shape: (1536,)

    # Dot product of unit vectors = cosine similarity
    sims = np.dot(emb_norm, q_norm)           # shape: (num_chunks,) — one score per chunk

    # Step 3: get the indices of the top-k most similar chunks
    top_idx = np.argsort(sims)[::-1][:k]      # sort descending, take first k

    # Step 4: return the text of those chunks, separated by "---"
    parts = [_chunks[i] for i in top_idx]
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Status helpers (used by /health endpoint in main.py)
# ---------------------------------------------------------------------------

def is_rag_ready() -> bool:
    """True if the RAG index was built successfully and is ready for queries."""
    return _rag_ready


def get_rag_error() -> str | None:
    """Error message from build_rag_index(), or None if successful."""
    return _rag_error


def get_num_documents() -> int:
    """Number of source documents loaded from pathologies/ (for /health info)."""
    return _num_documents


def get_num_chunks() -> int:
    """Total number of text chunks in the index (for /health info)."""
    return len(_chunks)
