import re
from dataclasses import dataclass

import tiktoken

from rag.loader import Document

# An approximation: the embedding model will tokenize slightly differently,
# but it is close enough to keep chunks inside a model's input window.
_enc = tiktoken.get_encoding("cl100k_base")
_SEP = _enc.encode("\n\n")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None


def _blocks(text: str) -> list[str]:
    """Split on blank lines. Paragraphs are the smallest unit worth keeping whole."""
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _split_oversized(tokens: list[int], budget: int) -> list[list[int]]:
    return [tokens[i : i + budget] for i in range(0, len(tokens), budget)]


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Pack paragraphs into chunks of at most `chunk_size` tokens.

    Each chunk repeats the last `overlap` tokens of the previous one, so a
    sentence sitting on a boundary still appears whole in one of them.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Cap oversized paragraphs so a carried overlap can never push a chunk past
    # chunk_size.
    budget = max(1, chunk_size - overlap - len(_SEP))
    pieces: list[list[int]] = []
    for block in _blocks(text):
        pieces.extend(_split_oversized(_enc.encode(block), budget))

    chunks: list[list[int]] = []
    current: list[int] = []
    carried = 0  # tokens in `current` that are overlap, not new content
    for piece in pieces:
        separator = _SEP if current else []
        if len(current) > carried and len(current) + len(separator) + len(piece) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            carried = len(current)
            separator = _SEP if current else []
        current = current + separator + piece
    if len(current) > carried:  # a tail of pure overlap would just be a duplicate
        chunks.append(current)

    return [_enc.decode(chunk).strip() for chunk in chunks]


def chunk_documents(
    docs: list[Document], chunk_size: int = 512, overlap: int = 64
) -> list[Chunk]:
    chunks = []
    for doc in docs:
        for index, text in enumerate(split_text(doc.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{doc.source}:{doc.page or 0}:{index}",
                    text=text,
                    source=doc.source,
                    page=doc.page,
                )
            )
    return chunks
