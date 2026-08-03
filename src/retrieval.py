"""课程级检索：向量检索 / BM25 / 混合检索（RRF）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.config import (
    DEFAULT_TOP_K,
    DEFAULT_USE_HYBRID,
    HYBRID_CANDIDATE_MULTIPLIER,
    RRF_K,
)
from src.embeddings import DashScopeEmbedder
from src.registry import CourseRegistry
from src.vector_store import CourseVectorStore, SearchHit


@dataclass
class RetrievedContext:
    """供生成使用的检索结果。"""

    hits: List[SearchHit]
    context_text: str
    references: List[dict]
    mode: str = "vector"


class CourseRetriever:
    """按课程路由的检索器，支持向量 / BM25 / 混合。"""

    def __init__(
        self,
        registry: Optional[CourseRegistry] = None,
        embedder: Optional[DashScopeEmbedder] = None,
    ):
        self.registry = registry or CourseRegistry()
        self.embedder = embedder or DashScopeEmbedder()

    def retrieve(
        self,
        course_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        expand_parent: bool = False,
        use_hybrid: bool = DEFAULT_USE_HYBRID,
        mode: Optional[str] = None,
    ) -> RetrievedContext:
        """
        mode:
          - hybrid: 向量 + BM25 RRF 融合（默认）
          - vector: 仅向量
          - bm25: 仅 BM25
        use_hybrid=True 且未指定 mode 时等价于 hybrid。
        """
        self.registry.get_course(course_id)
        store = CourseVectorStore(self.registry.vector_dir(course_id), embedder=self.embedder)

        if mode is None:
            mode = "hybrid" if use_hybrid else "vector"

        if mode == "hybrid":
            hits = store.search_hybrid(
                query,
                top_k=top_k,
                candidate_multiplier=HYBRID_CANDIDATE_MULTIPLIER,
                rrf_k=RRF_K,
            )
        elif mode == "bm25":
            hits = store.search_bm25(query, top_k=top_k)
        else:
            hits = store.search(query, top_k=top_k)
            mode = "vector"

        if expand_parent:
            hits = self._dedupe_parent_pages(hits)

        blocks: List[str] = []
        references: List[dict] = []
        seen_ref = set()
        for i, hit in enumerate(hits, start=1):
            body = hit.parent_text if expand_parent and hit.parent_text else hit.text
            label = f"[{i}] {hit.filename} · 第{hit.page}页/幻灯片"
            blocks.append(f"{label}\n{body}")
            key = (hit.filename, hit.page)
            if key not in seen_ref:
                seen_ref.add(key)
                references.append(
                    {
                        "filename": hit.filename,
                        "page": hit.page,
                        "score": round(hit.score, 4),
                        "snippet": hit.text[:200],
                    }
                )
        return RetrievedContext(
            hits=hits,
            context_text="\n\n---\n\n".join(blocks),
            references=references,
            mode=mode,
        )

    @staticmethod
    def _dedupe_parent_pages(hits: List[SearchHit]) -> List[SearchHit]:
        """父页扩展后按 (doc_id, page) 去重，保留最高分。"""
        best: dict[tuple, SearchHit] = {}
        for hit in hits:
            key = (hit.doc_id, hit.page)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        return sorted(best.values(), key=lambda h: h.score, reverse=True)
