"""Document loaders: markdown, text, and PDF (if pypdf is installed)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str      # filename stem, e.g. "vacation-policy"
    title: str
    text: str
    path: str


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_docs(docs_dir: Path) -> list[Document]:
    """Load every supported file under docs_dir, including subfolders.

    doc_id is the path relative to docs_dir without its extension
    ("vacation-policy", "hr/benefits"). If two files would share an id
    (notes.md + notes.txt), the later one keeps its extension ("notes.txt").
    """
    docs: list[Document] = []
    seen: set[str] = set()
    for path in sorted(docs_dir.rglob("*")):
        rel = path.relative_to(docs_dir)
        if not path.is_file() or any(p.startswith(".") for p in rel.parts):
            continue
        suffix = path.suffix.lower()
        doc_id = rel.with_suffix("").as_posix()
        if doc_id in seen:
            doc_id = rel.as_posix()
        if suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8")
            title = _title_from_markdown(text, doc_id) if suffix == ".md" else doc_id
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                print(f"  \u26a0 skipping {path.name}: pip install pypdf for PDF support")
                continue
            reader = PdfReader(str(path))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
            title = doc_id
        else:
            continue
        text = text.strip()
        if text:
            seen.add(doc_id)
            docs.append(Document(doc_id, title, text, str(path)))
    return docs
