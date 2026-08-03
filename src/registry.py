"""课程与课件元数据注册表（JSON 持久化）。"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config import DATA_DIR, REGISTRY_PATH, ensure_data_dirs


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(name: str) -> str:
    """从课程名生成可读短标识。"""
    text = name.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^\w\u4e00-\u9fff\-]+", "", text)
    return text[:40] or "course"


@dataclass
class DocumentMeta:
    """单份课件元数据。"""

    doc_id: str
    filename: str
    stored_path: str
    chunk_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


@dataclass
class CourseMeta:
    """单门课程元数据。"""

    course_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    documents: List[DocumentMeta] = field(default_factory=list)


class CourseRegistry:
    """维护课程列表与文档清单；课程目录在 data/courses/{course_id}/。"""

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        ensure_data_dirs()
        self.registry_path = registry_path
        self._courses: dict[str, CourseMeta] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self._courses = {}
            self._save()
            return
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self._courses = {}
        for item in raw.get("courses", []):
            docs = [DocumentMeta(**d) for d in item.get("documents", [])]
            course = CourseMeta(
                course_id=item["course_id"],
                name=item["name"],
                description=item.get("description", ""),
                created_at=item.get("created_at", _now_iso()),
                updated_at=item.get("updated_at", _now_iso()),
                documents=docs,
            )
            self._courses[course.course_id] = course

    def _save(self) -> None:
        payload = {
            "courses": [
                {
                    **{k: v for k, v in asdict(c).items() if k != "documents"},
                    "documents": [asdict(d) for d in c.documents],
                }
                for c in self._courses.values()
            ]
        }
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def course_dir(self, course_id: str) -> Path:
        return DATA_DIR / course_id

    def raw_dir(self, course_id: str) -> Path:
        return self.course_dir(course_id) / "raw"

    def vector_dir(self, course_id: str) -> Path:
        """课程 FAISS 向量库目录。"""
        return self.course_dir(course_id) / "faiss"

    # 兼容旧调用名
    def chroma_dir(self, course_id: str) -> Path:
        return self.vector_dir(course_id)

    def list_courses(self) -> List[CourseMeta]:
        return sorted(self._courses.values(), key=lambda c: c.created_at)

    def get_course(self, course_id: str) -> CourseMeta:
        if course_id not in self._courses:
            raise KeyError(f"课程不存在: {course_id}")
        return self._courses[course_id]

    def create_course(self, name: str, description: str = "") -> CourseMeta:
        name = name.strip()
        if not name:
            raise ValueError("课程名称不能为空")
        for c in self._courses.values():
            if c.name == name:
                raise ValueError(f"课程名称已存在: {name}")
        short = _slugify(name)
        course_id = f"{short}-{uuid.uuid4().hex[:8]}"
        course = CourseMeta(course_id=course_id, name=name, description=description.strip())
        self.raw_dir(course_id).mkdir(parents=True, exist_ok=True)
        self.vector_dir(course_id).mkdir(parents=True, exist_ok=True)
        self._courses[course_id] = course
        self._save()
        return course

    def rename_course(self, course_id: str, new_name: str, description: Optional[str] = None) -> CourseMeta:
        course = self.get_course(course_id)
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("课程名称不能为空")
        for c in self._courses.values():
            if c.course_id != course_id and c.name == new_name:
                raise ValueError(f"课程名称已存在: {new_name}")
        course.name = new_name
        if description is not None:
            course.description = description.strip()
        course.updated_at = _now_iso()
        self._save()
        return course

    def delete_course(self, course_id: str) -> None:
        self.get_course(course_id)
        del self._courses[course_id]
        self._save()
        course_path = self.course_dir(course_id)
        if course_path.exists():
            shutil.rmtree(course_path)

    def get_document(self, course_id: str, doc_id: str) -> DocumentMeta:
        course = self.get_course(course_id)
        for doc in course.documents:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(f"文档不存在: {doc_id}")

    def add_document(
        self,
        course_id: str,
        filename: str,
        stored_path: str,
        chunk_count: int,
        doc_id: Optional[str] = None,
    ) -> DocumentMeta:
        course = self.get_course(course_id)
        doc_id = doc_id or uuid.uuid4().hex
        # 同名文件视为更新：先移除旧记录
        course.documents = [d for d in course.documents if d.filename != filename and d.doc_id != doc_id]
        doc = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            stored_path=stored_path,
            chunk_count=chunk_count,
        )
        course.documents.append(doc)
        course.updated_at = _now_iso()
        self._save()
        return doc

    def remove_document(self, course_id: str, doc_id: str, delete_file: bool = True) -> DocumentMeta:
        course = self.get_course(course_id)
        doc = self.get_document(course_id, doc_id)
        course.documents = [d for d in course.documents if d.doc_id != doc_id]
        course.updated_at = _now_iso()
        self._save()
        if delete_file:
            path = Path(doc.stored_path)
            if path.exists():
                path.unlink()
        return doc

    def update_document_chunk_count(self, course_id: str, doc_id: str, chunk_count: int) -> None:
        doc = self.get_document(course_id, doc_id)
        doc.chunk_count = chunk_count
        doc.updated_at = _now_iso()
        self.get_course(course_id).updated_at = _now_iso()
        self._save()
