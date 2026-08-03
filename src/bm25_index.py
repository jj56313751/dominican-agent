"""课程级 BM25 索引（对齐 RAG-cy：rank_bm25 + pickle）。

索引与 chunks.json 行序一致；中文采用字/双字切分，便于课件关键词命中。
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

BM25_FILE = "bm25.pkl"


def tokenize(text: str) -> List[str]:
    """中英混合分词：英文按词，中文按单字+双字。"""
    if not text:
        return []
    text = text.lower()
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
    tokens: List[str] = []
    for part in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            chars = list(part)
            tokens.extend(chars)
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class CourseBM25Index:
    """与课程 chunks 对齐的 BM25Okapi 索引。"""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / BM25_FILE
        self._bm25: BM25Okapi | None = None
        self._size: int = 0

    @property
    def ready(self) -> bool:
        return self._bm25 is not None and self._size > 0

    def rebuild(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """根据当前 chunks 全量重建并落盘。"""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if not chunks:
            self._bm25 = None
            self._size = 0
            if self.path.exists():
                self.path.unlink()
            return
        corpus = [tokenize(str(c.get("text", ""))) for c in chunks]
        # 避免空文档导致 BM25 异常：至少放一个占位 token
        corpus = [toks if toks else ["_empty_"] for toks in corpus]
        self._bm25 = BM25Okapi(corpus)
        self._size = len(corpus)
        with open(self.path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "size": self._size}, f)

    def load_or_rebuild(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """优先加载已有索引；数量不一致则重建。"""
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    bm25 = payload.get("bm25")
                    size = int(payload.get("size", 0))
                else:
                    # 兼容仅 dump BM25Okapi 的格式
                    bm25 = payload
                    size = len(chunks)
                if bm25 is not None and size == len(chunks) and len(chunks) > 0:
                    self._bm25 = bm25
                    self._size = size
                    return
            except Exception:
                pass
        self.rebuild(chunks)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """返回 [(chunk_index, score), ...]，按分数降序。"""
        if not self.ready or top_k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
