"""首页。"""
from __future__ import annotations

import streamlit as st

from src.config import DASHSCOPE_API_KEY, SAMPLE_COURSE_DIR, ensure_data_dirs

ensure_data_dirs()

st.title("多课程 RAG 知识库")
st.markdown(
    """
面向课程课件的检索增强问答应用。

**使用步骤**
1. 在「课程管理」中新建课程  
2. 在「课件入库」中上传或导入课件并向量化  
3. 在「课程问答」中选择课程提问  

请使用左侧页面导航切换功能。
"""
)

if not DASHSCOPE_API_KEY:
    st.error("未检测到 DASHSCOPE_API_KEY。请复制 `.env.example` 为 `.env` 并填入密钥后重启应用。")
else:
    st.success("已加载 DASHSCOPE_API_KEY")

st.caption(f"样例课件目录：`{SAMPLE_COURSE_DIR}`")
