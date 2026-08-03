"""多课程 RAG 知识库 — 入口与中文侧边栏导航。"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="多课程 RAG 知识库",
    page_icon=None,
    layout="wide",
)

_VIEWS = Path(__file__).resolve().parent / "views"

pg = st.navigation(
    {
        "导航": [
            st.Page(_VIEWS / "home.py", title="首页", default=True),
            st.Page(_VIEWS / "course_management.py", title="课程管理"),
            st.Page(_VIEWS / "document_ingest.py", title="课件入库"),
            st.Page(_VIEWS / "course_qa.py", title="课程问答"),
        ]
    }
)
pg.run()
