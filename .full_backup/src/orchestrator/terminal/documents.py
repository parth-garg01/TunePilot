"""Document Context & Chunking Engine (PRD Section 10, Feature Update)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DocumentChunk:
    doc_name: str
    chunk_index: int
    text: str
    score: float = 0.0


class DocumentEngine:
    """Parses, chunks, and retrieves relevant snippets from documents (Markdown, TXT, PDF, DOCX)."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.indexed_chunks: list[DocumentChunk] = []

    def parse_document(self, path: Path | str) -> str:
        p = Path(path)
        if not p.exists():
            return ""

        ext = p.suffix.lower()
        if ext in {".md", ".txt", ".yaml", ".yml", ".json", ".py", ".sh"}:
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""

        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(p))
                return "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                return f"[PDF document: {p.name}]"

        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(str(p))
                return "\n\n".join(paragraph.text for paragraph in doc.paragraphs)
            except Exception:
                return f"[DOCX document: {p.name}]"

        return ""

    def chunk_text(self, doc_name: str, text: str) -> list[DocumentChunk]:
        if not text:
            return []

        # Split by paragraphs / double newlines first
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[DocumentChunk] = []
        current_buf: list[str] = []
        current_len = 0
        chunk_idx = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > self.chunk_size and current_buf:
                chunk_str = "\n\n".join(current_buf).strip()
                chunks.append(DocumentChunk(doc_name=doc_name, chunk_index=chunk_idx, text=chunk_str))
                chunk_idx += 1
                # Overlap: keep last paragraph if small
                if len(current_buf[-1]) < self.chunk_overlap:
                    current_buf = [current_buf[-1], p]
                    current_len = len(current_buf[0]) + p_len
                else:
                    current_buf = [p]
                    current_len = p_len
            else:
                current_buf.append(p)
                current_len += p_len

        if current_buf:
            chunks.append(
                DocumentChunk(
                    doc_name=doc_name,
                    chunk_index=chunk_idx,
                    text="\n\n".join(current_buf).strip(),
                )
            )

        return chunks

    def index_document(self, path: Path | str) -> int:
        p = Path(path)
        content = self.parse_document(p)
        chunks = self.chunk_text(p.name, content)
        self.indexed_chunks.extend(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        """Simple lexical TF-IDF / term-frequency search over indexed chunks."""
        if not self.indexed_chunks:
            return []

        terms = set(re.findall(r"\w+", query.lower()))
        if not terms:
            return self.indexed_chunks[:top_k]

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self.indexed_chunks:
            chunk_words = re.findall(r"\w+", chunk.text.lower())
            total_words = max(len(chunk_words), 1)
            score = 0.0
            for term in terms:
                term_count = chunk_words.count(term)
                if term_count > 0:
                    score += (term_count / total_words) * (len(term) ** 0.5)

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            c = DocumentChunk(
                doc_name=chunk.doc_name,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=score,
            )
            results.append(c)

        return results
