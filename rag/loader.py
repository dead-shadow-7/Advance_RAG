from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".md", ".txt"}


@dataclass
class Document:
    """One unit of source text. PDFs give one Document per page."""

    text: str
    source: str  # path relative to the data directory
    page: int | None = None


def load_pdf(path: Path, source: str) -> list[Document]:
    reader = PdfReader(path)
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # scanned pages extract to nothing; OCR is out of scope
            docs.append(Document(text=text, source=source, page=number))
    return docs


def load_text(path: Path, source: str) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(text=text, source=source)] if text else []


def load_directory(directory: str | Path = "data") -> list[Document]:
    directory = Path(directory)
    docs = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        source = path.relative_to(directory).as_posix()
        if path.suffix.lower() == ".pdf":
            docs.extend(load_pdf(path, source))
        else:
            docs.extend(load_text(path, source))
    return docs
