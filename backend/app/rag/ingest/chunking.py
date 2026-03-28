from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class ChunkInput:
    document_id: str
    title: str
    text: str
    url: str | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class ChunkRecord:
    point_id: str
    document_id: str
    chunk_index: int
    title: str
    text: str
    url: str | None
    metadata: dict[str, str]


def build_chunks(
    document: ChunkInput,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = [part.strip() for part in splitter.split_text(document.text) if part.strip()]
    metadata = document.metadata or {}
    return [
        ChunkRecord(
            point_id=str(uuid5(NAMESPACE_URL, f"{document.document_id}:{idx}")),
            document_id=document.document_id,
            chunk_index=idx,
            title=document.title,
            text=chunk_text,
            url=document.url,
            metadata=metadata,
        )
        for idx, chunk_text in enumerate(raw_chunks)
    ]
