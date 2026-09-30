from dataclasses import dataclass
import re

from core.document_loader import DocumentPage


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page_number: int | None
    text: str


def create_chunks(
    pages: list[DocumentPage],
    source: str,
    chunk_size: int = 180,
    overlap: int = 35,
) -> list[Chunk]:
    """Split each source page into overlapping word windows, preferring sentence ends."""
    if chunk_size < 20:
        raise ValueError("Chunk size must be at least 20 words.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            if end < len(words):
                for candidate in range(end, min(start + chunk_size + 1, len(words))):
                    if re.search(r"[.!?][\]\)'\"]?$", words[candidate - 1]):
                        end = candidate
                        break
            text = " ".join(words[start:end]).strip()
            if text:
                chunk_id = f"{_path_stem(source)}-{page.page_number or 'doc'}-{len(chunks) + 1:03d}"
                chunks.append(Chunk(chunk_id, source, page.page_number, text))
            if end >= len(words):
                break
            start = max(end - overlap, start + 1)
    return chunks


def _path_stem(source: str) -> str:
    """Produce a readable identifier prefix without exposing a path."""
    from pathlib import Path

    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(source).stem).strip("-")
    return stem or "document"
