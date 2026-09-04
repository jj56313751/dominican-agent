"""基于检索上下文的 DashScope Qwen 问答。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import dashscope
from dashscope import Generation, MultiModalConversation

from src.config import DASHSCOPE_API_KEY, DEFAULT_TOP_K, DEFAULT_USE_HYBRID, LLM_MODEL
from src.embeddings import DashScopeEmbedder
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.registry import CourseRegistry
from src.retrieval import CourseRetriever, RetrievedContext


def _uses_multimodal_endpoint(model: str) -> bool:
    """判断模型是否必须走 DashScope 多模态端点。

    qwen3.7-plus / qwen3.8-max / VL / Omni 等与 Generation（text-generation）
    不匹配，会返回 400 InvalidParameter: url error。
    qwen3.7-max、qwen-plus 等纯文本模型仍使用 Generation。
    """
    name = (model or "").strip().lower()
    if not name:
        return False
    if name.startswith("qwen3.7-max") or name.startswith("qwen3-max"):
        return False
    if name.startswith("qwen-plus") or name.startswith("qwen-turbo") or name.startswith("qwen-max"):
        return False
    if "vl" in name or "omni" in name:
        return True
    if name.startswith("qwen3.7-plus") or name.startswith("qwen3.8") or name.startswith("qwen3.6") or name.startswith("qwen3.5"):
        return True
    return False


def _extract_answer_text(content: Any) -> str:
    """从 Generation / MultiModalConversation 的 content 中取出纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _call_llm(model: str, system_prompt: str, user_content: str):
    """按模型类型选择 Generation 或 MultiModalConversation。"""
    if _uses_multimodal_endpoint(model):
        # 多模态模型要求 content 为 [{text: ...}]；非流式需关闭思考模式
        return MultiModalConversation.call(
            model=model,
            messages=[
                {"role": "system", "content": [{"text": system_prompt}]},
                {"role": "user", "content": [{"text": user_content}]},
            ],
            temperature=0.2,
            enable_thinking=False,
        )
    return Generation.call(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        result_format="message",
    )


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
        response = _call_llm(self.model, SYSTEM_PROMPT, user_content)
        if getattr(response, "status_code", 200) != 200:
            code = getattr(response, "code", None)
            message = getattr(response, "message", None)
            raise RuntimeError(
                f"LLM 调用失败: status={getattr(response, 'status_code', None)}, "
                f"code={code}, message={message}。"
                f"若为额度/权限问题，请在阿里云百炼控制台充值或开通对应模型后重试。"
            )
        content = _extract_answer_text(response.output.choices[0].message.content)
        return AnswerResult(
            answer=content,
            references=retrieved.references,
            context_text=retrieved.context_text,
            model=self.model,
            retrieval_mode=retrieved.mode,
        )
