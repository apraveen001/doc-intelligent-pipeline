import re
from io import BytesIO

import pdfplumber


# ── Constants ────────────────────────────────────────────────────────────────

CHUNK_SIZE = 500      # target words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF byte stream using pdfplumber."""
    text_parts = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

    return "\n\n".join(text_parts)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into chunks of ~chunk_size words with overlap.

    Returns a list of dicts:
        {
            "chunk_index": int,
            "text": str,
            "word_count": int,
        }
    """
    # Normalise whitespace but keep paragraph breaks as single newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    words = text.split()          # split on any whitespace

    if not words:
        return []

    chunks = []
    start = 0                     # word index

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            {
                "chunk_index": len(chunks),
                "text": chunk_text_str,
                "word_count": len(chunk_words),
            }
        )

        if end == len(words):
            break

        # Slide forward by (chunk_size - overlap) so the next chunk
        # re-uses the last `overlap` words of the current chunk.
        start += chunk_size - overlap

    return chunks


# ── Public entry point ────────────────────────────────────────────────────────

def parse_and_chunk(file_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline: bytes -> text -> chunks.

    Returns:
        {
            "filename": str,
            "total_words": int,
            "total_chunks": int,
            "chunks": [ {"chunk_index": int, "text": str, "word_count": int}, ... ]
        }
    """
    raw_text = extract_text_from_pdf(file_bytes)

    if not raw_text.strip():
        return {
            "filename": filename,
            "total_words": 0,
            "total_chunks": 0,
            "chunks": [],
            "warning": "No extractable text found. The PDF may be scanned/image-based.",
        }

    chunks = chunk_text(raw_text)

    return {
        "filename": filename,
        "total_words": len(raw_text.split()),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }