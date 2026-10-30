"""基于检索上下文的 DashScope Qwen 问答。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import dashscope
from dashscope import Generation, MultiModalConversation

from src.config import DASHSCOPE_API_KEY, DEFAULT_TOP_K, DEFAULT_USE_HYBRID, LLM_MODEL
from src.embeddings import DashScopeEmbedder
from src.prompts import (
    COURSEWARE_INSUFFICIENT_MARK,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    WEB_SYSTEM_PROMPT,
    WEB_USER_TEMPLATE,
)
from src.registry import CourseRegistry
from src.retrieval import CourseRetriever, RetrievedContext


def _uses_multimodal_endpoint(model: str) -> bool:
    """判断模型是否必须走 DashScope 多模态端点。

    qwen3.7-plus / qwen3.7-flash / qwen3.8-max / VL / Omni 等与 Generation
    （text-generation）不匹配，会返回 400 InvalidParameter: url error。
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
    if (
        name.startswith("qwen3.7-plus")
        or name.startswith("qwen3.7-flash")
        or name.startswith("qwen3.8")
        or name.startswith("qwen3.6")
        or name.startswith("qwen3.5")
    ):
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


def _search_strategy(model: str) -> str:
    """按模型选择联网搜索策略。qwen3.8 多模态接口不支持 agent。"""
    name = (model or "").strip().lower()
    if "omni" in name:
        return "agent"
    return "max"


def _raise_if_llm_failed(response) -> None:
    """沿用原有失败信息：非 200 时抛出额度或参数错误。"""
    if getattr(response, "status_code", 200) != 200:
        code = getattr(response, "code", None)
        message = getattr(response, "message", None)
        raise RuntimeError(
            f"LLM 调用失败: status={getattr(response, 'status_code', None)}, "
            f"code={code}, message={message}。"
            f"若为额度/权限问题，请在阿里云百炼控制台充值或开通对应模型后重试。"
        )


def _courseware_insufficient(answer: str) -> bool:
    """课件答案是否明确表示上下文不够。"""
    return COURSEWARE_INSUFFICIENT_MARK in (answer or "")


def _normalize_web_references(results: Any) -> List[dict]:
    """把百炼 search_results 收成标题、链接和编号。"""
    if not results:
        return []
    refs: List[dict] = []
    seen = set()
    for item in results:
        if isinstance(item, dict):
            title = item.get("title") or ""
            url = item.get("url") or ""
            index = item.get("index")
            site_name = item.get("site_name") or ""
        else:
            title = getattr(item, "title", "") or ""
            url = getattr(item, "url", "") or ""
            index = getattr(item, "index", None)
            site_name = getattr(item, "site_name", "") or ""
        title = str(title).strip()
        url = str(url).strip()
        site_name = str(site_name).strip()
        if not title and not url:
            continue
        key = url or title
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            {
                "index": index,
                "title": title,
                "url": url,
                "site_name": site_name,
            }
        )
    return refs


def _search_results_from_output(output: Any) -> List[dict]:
    """从单次响应的 output.search_info 取出搜索来源。"""
    if output is None:
        return []
    if isinstance(output, dict):
        info = output.get("search_info")
    else:
        info = getattr(output, "search_info", None)
    if not info:
        return []
    if isinstance(info, dict):
        results = info.get("search_results") or []
    else:
        results = getattr(info, "search_results", None) or []
    return _normalize_web_references(results)


def _merge_stream_text(parts: List[str]) -> str:
    """拼接流式文本。后续片段若以已有文本开头，视为累计输出并替换。"""
    merged = ""
    for part in parts:
        if not part:
            continue
        if not merged or part.startswith(merged):
            merged = part
        elif merged.endswith(part):
            continue
        else:
            merged += part
    return merged


def _consume_search_stream(stream: Iterable) -> tuple[str, List[dict]]:
    """消费联网搜索流：多模态接口必须流式，否则会拒绝 Web Search。"""
    parts: List[str] = []
    web_references: List[dict] = []
    for chunk in stream:
        _raise_if_llm_failed(chunk)
        output = getattr(chunk, "output", None)
        found = _search_results_from_output(output)
        if found:
            web_references = found
        choices = None
        if isinstance(output, dict):
            choices = output.get("choices")
        elif output is not None:
            choices = getattr(output, "choices", None)
        if not choices:
            continue
        message = choices[0].message
        content = message.get("content") if isinstance(message, dict) else message.content
        text = _extract_answer_text(content)
        if text:
            parts.append(text)
    return _merge_stream_text(parts), web_references


def _call_llm_web_search(model: str, system_prompt: str, user_content: str):
    """课件不足时的第二次调用。只在这一次打开联网搜索。"""
    search_options = {
        "search_strategy": _search_strategy(model),
        "enable_source": True,
        "enable_citation": True,
        "citation_format": "[]",
    }
    if _uses_multimodal_endpoint(model):
        # qwen3.8-flash 等必须走多模态接口，且联网搜索只接受流式
        stream = MultiModalConversation.call(
            model=model,
            messages=[
                {"role": "system", "content": [{"text": system_prompt}]},
                {"role": "user", "content": [{"text": user_content}]},
            ],
            temperature=0.2,
            enable_thinking=False,
            enable_search=True,
            search_options=search_options,
            stream=True,
            incremental_output=True,
        )
        return _consume_search_stream(stream)

    response = Generation.call(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        result_format="message",
        enable_search=True,
        search_options=search_options,
    )
    _raise_if_llm_failed(response)
    content = _extract_answer_text(response.output.choices[0].message.content)
    return content, _search_results_from_output(response.output)


@dataclass
class AnswerResult:
    """问答结果。"""

    answer: str
    references: List[dict]
    context_text: str
    model: str
    retrieval_mode: str = "vector"
    # courseware：答案来自课件；web：课件不足后改用联网搜索
    answer_source: str = "courseware"
    web_references: List[dict] = field(default_factory=list)


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
        # 有命中时仍先只根据课件回答，这次调用不打开联网搜索
        if retrieved.hits:
            user_content = USER_TEMPLATE.format(
                course_name=course.name,
                context=retrieved.context_text,
                question=question,
            )
            response = _call_llm(self.model, SYSTEM_PROMPT, user_content)
            _raise_if_llm_failed(response)
            content = _extract_answer_text(response.output.choices[0].message.content)
            if not _courseware_insufficient(content):
                return AnswerResult(
                    answer=content,
                    references=retrieved.references,
                    context_text=retrieved.context_text,
                    model=self.model,
                    retrieval_mode=retrieved.mode,
                    answer_source="courseware",
                )

        # 未命中，或课件答案写明无法确定时，才联网搜索
        web_answer, web_references = _call_llm_web_search(
            self.model,
            WEB_SYSTEM_PROMPT,
            WEB_USER_TEMPLATE.format(course_name=course.name, question=question),
        )
        if not (web_answer or "").strip():
            raise RuntimeError("联网搜索未返回回答内容")
        return AnswerResult(
            answer=web_answer,
            references=retrieved.references,
            context_text=retrieved.context_text,
            model=self.model,
            retrieval_mode=retrieved.mode,
            answer_source="web",
            web_references=web_references,
        )
