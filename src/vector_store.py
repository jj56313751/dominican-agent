"""每门课程独立的 FAISS 向量库，支持按 doc_id 增删改。

FAISS 只存向量；分块原文与元数据存在 chunks.json。
行号 i 与 chunks[i] 一一对应。删除/更新时过滤后重建 IndexFlatIP。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import faiss
import numpy as np

from src.bm25_index import CourseBM25Index
from src.embeddings import DashScopeEmbedder

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


@dataclass
class ChunkRecord:
    """待写入向量库的一条分块。"""

    chunk_id: str
    text: str
    doc_id: str
    filename: str
    page: int
    parent_text: str
    chunk_index: int


@dataclass
class SearchHit:
    """检索命中结果。"""

    chunk_id: str
    text: str
    score: float
    doc_id: str
    filename: str
    page: int
    parent_text: str
    chunk_index: int


class CourseVectorStore:
    """课程级 FAISS 持久化库（IndexFlatIP + L2 归一化 ≈ 余弦相似度）。"""

    def __init__(self, store_dir: Path, embedder: Optional[DashScopeEmbedder] = None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or DashScopeEmbedder()
        self._index_path = self.store_dir / INDEX_FILE
        self._chunks_path = self.store_dir / CHUNKS_FILE
        self._chunks: List[Dict[str, Any]] = []
        self._index: Optional[faiss.Index] = None
        self._bm25 = CourseBM25Index(self.store_dir)
        self._load()

    def _load(self) -> None:
        if self._chunks_path.exists():
            self._chunks = json.loads(self._chunks_path.read_text(encoding="utf-8"))
        else:
            self._chunks = []
        if self._index_path.exists() and self._chunks:
            self._index = faiss.read_index(str(self._index_path))
            if self._index.ntotal != len(self._chunks):
                raise RuntimeError(
                    f"FAISS 与 chunks 数量不一致: index={self._index.ntotal}, chunks={len(self._chunks)}"
                )
        else:
            self._index = None
        # BM25 与 chunks 对齐（缺失或不一致则重建）
        self._bm25.load_or_rebuild(self._chunks)

    def _save(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_path.write_text(
            json.dumps(self._chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self._index is not None and self._index.ntotal > 0:
            faiss.write_index(self._index, str(self._index_path))
        elif self._index_path.exists():
            self._index_path.unlink()
        self._bm25.rebuild(self._chunks)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 归一化，使内积等价于余弦相似度。"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def _rebuild_from_embeddings(self, embeddings: List[List[float]]) -> None:
        if not embeddings:
            self._index = None
            return
        arr = self._normalize(np.array(embeddings, dtype=np.float32))
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        self._index = index

    def count(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        """向本课程向量库追加分块。"""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            item = asdict(chunk)
            item["parent_text"] = (item.get("parent_text") or "")[:8000]
            item["embedding"] = emb
            self._chunks.append(item)

        all_emb = [c["embedding"] for c in self._chunks]
        self._rebuild_from_embeddings(all_emb)
        self._save()
        return len(chunks)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档 ID 删除该文档全部向量并重建索引。"""
        kept = [c for c in self._chunks if c.get("doc_id") != doc_id]
        if len(kept) == len(self._chunks):
            return
        self._chunks = kept
        if self._chunks:
            self._rebuild_from_embeddings([c["embedding"] for c in self._chunks])
        else:
            self._index = None
        self._save()

    def replace_doc_chunks(self, doc_id: str, chunks: Sequence[ChunkRecord]) -> int:
        """更新文档：先删旧向量再写入新分块。"""
        self.delete_by_doc_id(doc_id)
        return self.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """语义检索。"""
        if not self._chunks or self._index is None or self._index.ntotal == 0:
            return []
        query_vec = np.array([self.embedder.embed_query(query)], dtype=np.float32)
        query_vec = self._normalize(query_vec)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)
        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(self._hit_from_index(int(idx), float(score)))
        return hits

    def search_bm25(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """BM25 关键词检索。"""
        if not self._chunks:
            return []
        if not self._bm25.ready:
            self._bm25.load_or_rebuild(self._chunks)
        results = self._bm25.search(query, top_k=top_k)
        return [self._hit_from_index(idx, score) for idx, score in results]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 3,
        rrf_k: int = 60,
    ) -> List[SearchHit]:
        """向量 + BM25 混合检索（RRF 融合，思想对齐企业知识库文档中的 hybrid search）。"""
        if not self._chunks:
            return []
        pool = max(top_k * candidate_multiplier, top_k)
        pool = min(pool, len(self._chunks))

        vector_hits = self.search(query, top_k=pool)
        bm25_hits = self.search_bm25(query, top_k=pool)

        # Reciprocal Rank Fusion
        fused: Dict[str, float] = {}
        hit_map: Dict[str, SearchHit] = {}
        for rank, hit in enumerate(vector_hits, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            hit_map[hit.chunk_id] = hit
        for rank, hit in enumerate(bm25_hits, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if hit.chunk_id not in hit_map:
                hit_map[hit.chunk_id] = hit

        ranked_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)[:top_k]
        results: List[SearchHit] = []
        for cid in ranked_ids:
            base = hit_map[cid]
            results.append(
                SearchHit(
                    chunk_id=base.chunk_id,
                    text=base.text,
                    score=float(fused[cid]),
                    doc_id=base.doc_id,
                    filename=base.filename,
                    page=base.page,
                    parent_text=base.parent_text,
                    chunk_index=base.chunk_index,
                )
            )
        return results

    def _hit_from_index(self, idx: int, score: float) -> SearchHit:
        meta = self._chunks[idx]
        return SearchHit(
            chunk_id=str(meta.get("chunk_id", "")),
            text=str(meta.get("text", "")),
            score=score,
            doc_id=str(meta.get("doc_id", "")),
            filename=str(meta.get("filename", "")),
            page=int(meta.get("page", 0)),
            parent_text=str(meta.get("parent_text", "")),
            chunk_index=int(meta.get("chunk_index", 0)),
        )

    def list_doc_ids(self) -> List[str]:
        """返回库中出现过的全部 doc_id。"""
        return sorted({str(c.get("doc_id")) for c in self._chunks if c.get("doc_id")})
