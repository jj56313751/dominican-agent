"""Streamlit 会话内共享服务实例。"""
from __future__ import annotations

import streamlit as st

from src.ingest import IngestService
from src.registry import CourseRegistry


def get_registry() -> CourseRegistry:
    if "registry" not in st.session_state:
        st.session_state.registry = CourseRegistry()
    return st.session_state.registry


def get_ingest() -> IngestService:
    if "ingest" not in st.session_state:
        st.session_state.ingest = IngestService(registry=get_registry())
    return st.session_state.ingest


def refresh_registry() -> CourseRegistry:
    """强制从磁盘重新加载注册表。"""
    st.session_state.registry = CourseRegistry()
    if "ingest" in st.session_state:
        st.session_state.ingest = IngestService(registry=st.session_state.registry)
    return st.session_state.registry
