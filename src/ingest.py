"""课件入库流水线：解析 → 分块 → 写入课程向量库。"""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.chunking import chunk_pages
from src.config import SAMPLE_COURSE_DIR, SAMPLE_COURSE_NAME, SUPPORTED_EXTENSIONS
from src.embeddings import DashScopeEmbedder
from src.parsing import parse_file
from src.registry import CourseMeta, CourseRegistry
from src.vector_store import CourseVectorStore


@dataclass
class IngestResult:
    """单文件入库结果。"""

    filename: str
    doc_id: str
    chunk_count: int
    ok: bool
    error: str = ""


class IngestService:
    """面向课程的文档增删改服务。"""

    def __init__(
        self,
        registry: Optional[CourseRegistry] = None,
        embedder: Optional[DashScopeEmbedder] = None,
    ):
        self.registry = registry or CourseRegistry()
        self.embedder = embedder or DashScopeEmbedder()

    def _store(self, course_id: str) -> CourseVectorStore:
        return CourseVectorStore(self.registry.vector_dir(course_id), embedder=self.embedder)

    def ingest_file(self, course_id: str, source_path: Path, replace_same_name: bool = True) -> IngestResult:
        """将单个文件学习并向量化入库。"""
        source_path = Path(source_path)
        filename = source_path.name
        try:
            course = self.registry.get_course(course_id)
            raw_dir = self.registry.raw_dir(course_id)
            raw_dir.mkdir(parents=True, exist_ok=True)
            dest = raw_dir / filename

            # 同名覆盖：先删向量与注册项；若目标路径即旧文件则暂不删盘上文件
            if replace_same_name:
                for doc in list(course.documents):
                    if doc.filename == filename:
                        self._store(course_id).delete_by_doc_id(doc.doc_id)
                        same_path = Path(doc.stored_path).resolve() == dest.resolve()
                        self.registry.remove_document(
                            course_id, doc.doc_id, delete_file=not same_path
                        )

            if source_path.resolve() != dest.resolve():
                shutil.copy2(source_path, dest)
            elif not dest.exists():
                raise FileNotFoundError(f"待入库文件不存在: {dest}")

            doc_id = uuid.uuid4().hex
            pages = parse_file(dest)
            chunks = chunk_pages(pages, doc_id=doc_id, filename=filename)
            if not chunks:
                raise ValueError("分块结果为空")

            store = self._store(course_id)
            store.add_chunks(chunks)
            self.registry.add_document(
                course_id=course_id,
                filename=filename,
                stored_path=str(dest),
                chunk_count=len(chunks),
                doc_id=doc_id,
            )
            return IngestResult(filename=filename, doc_id=doc_id, chunk_count=len(chunks), ok=True)
        except Exception as exc:
            return IngestResult(
                filename=filename,
                doc_id="",
                chunk_count=0,
                ok=False,
                error=str(exc),
            )

    def ingest_bytes(
        self,
        course_id: str,
        filename: str,
        data: bytes,
        replace_same_name: bool = True,
    ) -> IngestResult:
        """从上传字节写入临时文件后入库。"""
        raw_dir = self.registry.raw_dir(course_id)
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / filename
        dest.write_bytes(data)
        return self.ingest_file(course_id, dest, replace_same_name=replace_same_name)

    def ingest_paths(self, course_id: str, paths: Sequence[Path]) -> List[IngestResult]:
        results: List[IngestResult] = []
        for path in paths:
            results.append(self.ingest_file(course_id, path))
        return results

    def delete_document(self, course_id: str, doc_id: str) -> None:
        """删除课件：向量 + 原始文件 + 注册表。"""
        self._store(course_id).delete_by_doc_id(doc_id)
        self.registry.remove_document(course_id, doc_id)

    def reingest_document(self, course_id: str, doc_id: str) -> IngestResult:
        """重新学习已有课件（改：删旧向量再入库）。"""
        doc = self.registry.get_document(course_id, doc_id)
        path = Path(doc.stored_path)
        if not path.exists():
            raise FileNotFoundError(f"原始文件丢失: {path}")
        # 保留同一 doc_id 与文件名
        self._store(course_id).delete_by_doc_id(doc_id)
        pages = parse_file(path)
        chunks = chunk_pages(pages, doc_id=doc_id, filename=doc.filename)
        self._store(course_id).add_chunks(chunks)
        self.registry.update_document_chunk_count(course_id, doc_id, len(chunks))
        return IngestResult(
            filename=doc.filename,
            doc_id=doc_id,
            chunk_count=len(chunks),
            ok=True,
        )

    def ensure_sample_course(self) -> CourseMeta:
        """确保样例课程存在；若不存在则创建。"""
        for c in self.registry.list_courses():
            if c.name == SAMPLE_COURSE_NAME:
                return c
        return self.registry.create_course(
            SAMPLE_COURSE_NAME,
            description="样例课件知识库",
        )

    def import_sample_directory(self, course_id: Optional[str] = None) -> List[IngestResult]:
        """从样例课件目录一键导入。"""
        sample_dir = Path(SAMPLE_COURSE_DIR)
        if not sample_dir.exists():
            raise FileNotFoundError(f"样例目录不存在: {sample_dir}")
        if course_id is None:
            course_id = self.ensure_sample_course().course_id
        else:
            self.registry.get_course(course_id)

        paths = sorted(
            p
            for p in sample_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        # 去重：同名（去掉时间戳后缀变体）时保留较短/无数字后缀的一份较难，这里按文件名全量导入，同名覆盖
        return self.ingest_paths(course_id, paths)


def iter_supported_files(directory: Path) -> Iterable[Path]:
    directory = Path(directory)
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
