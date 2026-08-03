"""课程问答页。"""
from __future__ import annotations

import streamlit as st

from src.answering import AnswerService
from src.config import DEFAULT_TOP_K, DEFAULT_USE_HYBRID, EMBEDDING_MODEL, LLM_MODEL, ensure_data_dirs
from src.ui_state import get_registry

ensure_data_dirs()

st.title("课程问答")
registry = get_registry()
courses = registry.list_courses()
if not courses:
    st.warning("请先创建课程并入库课件。")
    st.stop()

course_labels = {f"{c.name} ({len(c.documents)} 份课件)": c.course_id for c in courses}
selected_label = st.selectbox("选择课程", list(course_labels.keys()))
course_id = course_labels[selected_label]
course = registry.get_course(course_id)

if not course.documents:
    st.warning("该课程尚无课件，请先到「课件入库」学习文档。")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    top_k = st.slider("检索条数 top_k", min_value=1, max_value=15, value=DEFAULT_TOP_K)
with col_b:
    expand_parent = st.checkbox("扩展为同页/同幻灯片全文（Parent Retrieval）", value=False)

retrieval_mode = st.radio(
    "检索方式",
    options=["hybrid", "vector", "bm25"],
    format_func=lambda m: {
        "hybrid": "混合检索（向量 + BM25 / RRF）",
        "vector": "仅向量检索",
        "bm25": "仅 BM25 关键词检索",
    }[m],
    index=0 if DEFAULT_USE_HYBRID else 1,
    horizontal=True,
)

question = st.text_area("输入问题", placeholder="例如：什么是情境领导理论？", height=100)

if st.button("生成答案", type="primary", disabled=not question.strip()):
    try:
        with st.spinner(f"检索（{retrieval_mode}）并调用 {LLM_MODEL}…"):
            service = AnswerService(registry=registry)
            result = service.answer(
                course_id=course_id,
                question=question,
                top_k=top_k,
                expand_parent=expand_parent,
                retrieval_mode=retrieval_mode,
            )
        st.subheader("答案")
        st.markdown(result.answer)
        st.caption(
            f"Embedding：{EMBEDDING_MODEL} · LLM：{result.model} · 检索：{result.retrieval_mode}"
        )

        st.subheader("引用")
        if not result.references:
            st.write("无引用")
        else:
            for i, ref in enumerate(result.references, start=1):
                with st.expander(
                    f"[{i}] {ref['filename']} · 第{ref['page']}页/幻灯片 · score={ref['score']}"
                ):
                    st.write(ref.get("snippet", ""))

        with st.expander("查看检索上下文"):
            st.text(result.context_text)
    except Exception as exc:
        st.error(str(exc))
