"""文本分块：在父页/幻灯片内按字数切分。"""
from __future__ import annotations

from typing import List
from uuid import uuid4

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.parsing import PageUnit
from src.vector_store import ChunkRecord


def chunk_pages(
    pages: List[PageUnit],
    *,
    doc_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[ChunkRecord]:
    """将页/幻灯片列表切成 ChunkRecord。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

    records: List[ChunkRecord] = []
    global_index = 0
    for page in pages:
        parent = page.text.strip()
        if not parent:
            continue
        pieces = _split_text(parent, chunk_size, chunk_overlap)
        for piece in pieces:
            records.append(
                ChunkRecord(
                    chunk_id=f"{doc_id}_{global_index}_{uuid4().hex[:8]}",
                    text=piece,
                    doc_id=doc_id,
                    filename=filename,
                    page=page.page,
                    parent_text=parent,
                    chunk_index=global_index,
                )
            )
            global_index += 1
    return records


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """按字符窗口切分，保留重叠。"""
    if len(text) <= chunk_size:
        return [text]
    parts: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return [p for p in parts if p]
