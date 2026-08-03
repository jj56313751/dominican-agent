"""基于检索上下文的 DashScope Qwen 问答。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import dashscope

from src.config import DASHSCOPE_API_KEY, DEFAULT_TOP_K, DEFAULT_USE_HYBRID, LLM_MODEL
from src.embeddings import DashScopeEmbedder
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.registry import CourseRegistry
from src.retrieval import CourseRetriever, RetrievedContext


@dataclass
class AnswerResult:
    """问答结果。"""

    answer: str
    references: List[dict]
    context_text: str
    model: str
    retrieval_mode: str = "vector"


class AnswerService:
    """课程 RAG 问答服务。"""

    def __init__(
        self,
        registry: Optional[CourseRegistry] = None,
        embedder: Optional[DashScopeEmbedder] = None,
        model: Optional[str] = None,
    ):
        key = DASHSCOPE_API_KEY
        if not key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，请在 .env 中设置")
        dashscope.api_key = key
        self.registry = registry or CourseRegistry()
        self.embedder = embedder or DashScopeEmbedder()
        self.retriever = CourseRetriever(registry=self.registry, embedder=self.embedder)
        self.model = model or LLM_MODEL

    def answer(
        self,
        course_id: str,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        expand_parent: bool = False,
        use_hybrid: bool = DEFAULT_USE_HYBRID,
        retrieval_mode: Optional[str] = None,
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        course = self.registry.get_course(course_id)
        retrieved: RetrievedContext = self.retriever.retrieve(
            course_id=course_id,
            query=question,
            top_k=top_k,
            expand_parent=expand_parent,
            use_hybrid=use_hybrid,
            mode=retrieval_mode,
        )
        if not retrieved.hits:
            return AnswerResult(
                answer="当前课程知识库为空或未检索到相关内容，请先入库课件后再提问。",
                references=[],
                context_text="",
                model=self.model,
                retrieval_mode=retrieved.mode,
            )

        user_content = USER_TEMPLATE.format(
            course_name=course.name,
            context=retrieved.context_text,
            question=question,
        )
        response = dashscope.Generation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            result_format="message",
        )
        if getattr(response, "status_code", 200) != 200:
            code = getattr(response, "code", None)
            message = getattr(response, "message", None)
            raise RuntimeError(
                f"LLM 调用失败: status={getattr(response, 'status_code', None)}, "
                f"code={code}, message={message}。"
                f"若为额度/权限问题，请在阿里云百炼控制台充值或开通对应模型后重试。"
            )
        content = response.output.choices[0].message.content
        return AnswerResult(
            answer=content,
            references=retrieved.references,
            context_text=retrieved.context_text,
            model=self.model,
            retrieval_mode=retrieved.mode,
        )
