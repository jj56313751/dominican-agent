"""DashScope Embedding 封装。"""
from __future__ import annotations

from typing import List, Sequence, Union

import dashscope
from dashscope import TextEmbedding

from src.config import DASHSCOPE_API_KEY, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class DashScopeEmbedder:
    """调用 DashScope TextEmbedding 生成向量。"""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or DASHSCOPE_API_KEY
        if not key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，请在 .env 中设置")
        dashscope.api_key = key
        self.model = model or EMBEDDING_MODEL

    def embed(self, texts: Union[str, Sequence[str]]) -> List[List[float]]:
        """对单条或多条文本做 embedding，自动分批。"""
        if isinstance(texts, str):
            chunks = [texts]
        else:
            chunks = list(texts)
        if not chunks:
            raise ValueError("待嵌入文本列表为空")
        if not all(isinstance(x, str) for x in chunks):
            raise ValueError("所有待嵌入文本必须为字符串")
        cleaned = [x.strip() for x in chunks]
        if any(not x for x in cleaned):
            raise ValueError("存在空字符串，无法做 embedding")

        all_vectors: List[List[float]] = []
        for i in range(0, len(cleaned), EMBEDDING_BATCH_SIZE):
            batch = cleaned[i : i + EMBEDDING_BATCH_SIZE]
            resp = TextEmbedding.call(model=self.model, input=batch)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Embedding 调用失败: status={resp.status_code}, code={getattr(resp, 'code', None)}, "
                    f"message={getattr(resp, 'message', None)}"
                )
            output = resp.output
            # 兼容 dict / 对象
            embeddings = output["embeddings"] if isinstance(output, dict) else output.embeddings
            # 按 text_index 排序，保证与输入顺序一致
            ordered = sorted(embeddings, key=lambda e: e["text_index"] if isinstance(e, dict) else e.text_index)
            for item in ordered:
                vec = item["embedding"] if isinstance(item, dict) else item.embedding
                all_vectors.append(list(vec))
        if len(all_vectors) != len(cleaned):
            raise RuntimeError(
                f"Embedding 数量与输入不一致: got={len(all_vectors)}, expect={len(cleaned)}"
            )
        return all_vectors

    def embed_query(self, query: str) -> List[float]:
        return self.embed(query)[0]
