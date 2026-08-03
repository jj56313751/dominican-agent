"""全局配置：路径、模型与分块参数。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 课程数据根目录
DATA_DIR = PROJECT_ROOT / "data" / "courses"
REGISTRY_PATH = DATA_DIR / "registry.json"

# 样例课件根目录与默认样例课
SAMPLE_COURSE_ROOT = PROJECT_ROOT.parent / "课件"
SAMPLE_COURSE_DIR = SAMPLE_COURSE_ROOT / "组织绩效的战略领导力"
SAMPLE_COURSE_NAME = "组织绩效的战略领导力"

# DashScope
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-max")

# Embedding 批大小（新模型单批建议更小）
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

# 分块：按父页/幻灯片内再切
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# 检索默认参数
DEFAULT_TOP_K = 5

# BM25 混合检索
DEFAULT_USE_HYBRID = True
# 混合时各路召回候选数倍率（再经 RRF 融合截断到 top_k）
HYBRID_CANDIDATE_MULTIPLIER = 3
# RRF 常数
RRF_K = 60

# 支持的课件扩展名
SUPPORTED_EXTENSIONS = {".pptx", ".docx", ".pdf", ".md", ".txt"}


def ensure_data_dirs() -> None:
    """确保数据根目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
