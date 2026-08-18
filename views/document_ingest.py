"""课件入库：上传、样例导入、删除与重新学习。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import (
    DATA_DIR,
    SAMPLE_COURSE_DIR,
    SAMPLE_COURSE_NAME,
    SAMPLE_COURSE_ROOT,
    SUPPORTED_EXTENSIONS,
    ensure_data_dirs,
)
from src.folder_picker import list_subdirectories, pick_directory
from src.ui_state import get_ingest, get_registry, refresh_registry

ensure_data_dirs()

st.title("课件入库")
registry = get_registry()
ingest = get_ingest()
courses = registry.list_courses()

st.caption(f"支持格式：{', '.join(sorted(SUPPORTED_EXTENSIONS))}")

tab_upload, tab_sample, tab_manage = st.tabs(["上传课件", "样例一键导入", "已入库课件"])

with tab_upload:
    if not courses:
        st.warning("请先在「课程管理」中创建课程，或使用「样例一键导入」自动建课。")
    else:
        course_labels = {f"{c.name} ({c.course_id})": c.course_id for c in courses}
        selected_label = st.selectbox("选择课程", list(course_labels.keys()), key="upload_course")
        course_id = course_labels[selected_label]

        uploads = st.file_uploader(
            "选择一个或多个课件文件",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        if st.button("开始学习并向量化", type="primary", disabled=not uploads):
            progress = st.progress(0.0)
            status = st.empty()
            results = []
            for i, f in enumerate(uploads):
                status.write(f"正在处理：{f.name}")
                result = ingest.ingest_bytes(course_id, f.name, f.getvalue())
                results.append(result)
                progress.progress((i + 1) / len(uploads))
            refresh_registry()
            ok_n = sum(1 for r in results if r.ok)
            st.success(f"完成：成功 {ok_n} / {len(results)}")
            for r in results:
                if r.ok:
                    st.write(f"- {r.filename} → {r.chunk_count} chunks")
                else:
                    st.error(f"- {r.filename} 失败：{r.error}")

with tab_sample:
    if st.session_state.get("last_import_summary"):
        st.success(st.session_state.pop("last_import_summary"))
        for line in st.session_state.pop("last_import_lines", []):
            st.write(line)
        for err in st.session_state.pop("last_import_errors", []):
            st.error(err)

    st.markdown("#### 1. 选择数据源目录")
    st.caption(f"课程向量数据将写入：`{DATA_DIR}/{{course_id}}/`（与源目录分离）")

    if "sample_source_dir" not in st.session_state:
        default = SAMPLE_COURSE_DIR if SAMPLE_COURSE_DIR.exists() else SAMPLE_COURSE_ROOT
        st.session_state["sample_source_dir"] = str(default) if default.exists() else ""

    # 快捷选择：课件根目录下的子文件夹
    known_dirs = list_subdirectories(SAMPLE_COURSE_ROOT)
    known_labels = {p.name: str(p) for p in known_dirs}

    # 浏览目录后不能在本轮改 sample_quick_dir（控件已实例化），
    # 标记放到下一轮 rerun、selectbox 创建之前再重置。
    if st.session_state.pop("_reset_sample_quick_dir", False):
        st.session_state["sample_quick_dir"] = "（从下方浏览选择）"

    col_quick, col_browse = st.columns([3, 1])
    with col_quick:
        quick_options = ["（从下方浏览选择）"] + list(known_labels.keys())
        current_path = st.session_state.get("sample_source_dir", "")
        # 若当前路径正好是已知子目录，默认选中该项
        current_name = Path(current_path).name if current_path else ""
        default_idx = 0
        if current_name in known_labels and known_labels[current_name] == current_path:
            default_idx = quick_options.index(current_name)

        picked_name = st.selectbox(
            "快捷选择（课件库子目录）",
            quick_options,
            index=default_idx,
            key="sample_quick_dir",
        )
        if picked_name != "（从下方浏览选择）":
            st.session_state["sample_source_dir"] = known_labels[picked_name]

    with col_browse:
        st.write("")
        st.write("")
        if st.button("浏览目录…", use_container_width=True):
            initial = st.session_state.get("sample_source_dir") or str(SAMPLE_COURSE_ROOT)
            try:
                chosen = pick_directory(initial_dir=initial)
            except Exception as exc:
                st.error(f"打开目录选择器失败：{exc}")
                chosen = None
            if chosen:
                st.session_state["sample_source_dir"] = chosen
                st.session_state["_reset_sample_quick_dir"] = True
                st.rerun()
            else:
                st.info("未选择目录（已取消）")

    source_dir_str = (st.session_state.get("sample_source_dir") or "").strip()
    if source_dir_str:
        st.code(source_dir_str, language=None)
    else:
        st.warning("尚未选择目录，请使用快捷选择或点击「浏览目录…」")

    source_dir = Path(source_dir_str).expanduser() if source_dir_str else None

    # 目录变更时，同步更新「新建课程」默认名称
    prev_source = st.session_state.get("_sample_source_dir_prev")
    current_source_key = str(source_dir) if source_dir else ""
    if prev_source != current_source_key:
        st.session_state["_sample_source_dir_prev"] = current_source_key
        if source_dir and source_dir.name:
            st.session_state["sample_new_course_name"] = source_dir.name

    source_ok = False
    source_files: list[Path] = []
    if not source_dir_str:
        pass
    elif source_dir is None or not source_dir.exists():
        st.error(f"目录不存在：`{source_dir_str}`")
    elif not source_dir.is_dir():
        st.error(f"路径不是目录：`{source_dir_str}`")
    else:
        source_files = sorted(
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        source_ok = True
        st.success(f"已选择目录，可导入文件数：**{len(source_files)}**")
        if source_files:
            with st.expander("查看文件列表"):
                for p in source_files:
                    st.write(f"- {p.name}")
        else:
            st.warning("该目录下没有支持的课件文件")

    st.markdown("#### 2. 导入目标课程")
    target_mode = st.radio(
        "导入目标",
        ["新建课程后导入", "导入到已有课程"],
        horizontal=True,
        key="sample_target_mode",
    )

    target_course_id = None
    new_course_name = ""

    if target_mode == "新建课程后导入":
        if "sample_new_course_name" not in st.session_state:
            default_name = source_dir.name if source_dir and source_dir.name else SAMPLE_COURSE_NAME
            st.session_state["sample_new_course_name"] = default_name
        new_course_name = st.text_input(
            "课程名称（必填）",
            key="sample_new_course_name",
            placeholder="请输入课程名称",
        ).strip()
        if not new_course_name:
            st.error("课程名称为必填项")
        else:
            existing_names = {c.name for c in courses}
            if new_course_name in existing_names:
                st.warning(f"课程名「{new_course_name}」已存在，导入时将自动追加时间后缀。")
    else:
        if not courses:
            st.warning("当前还没有课程，请改选「新建课程后导入」。")
        else:
            course_labels = {f"{c.name} ({c.course_id})": c.course_id for c in courses}
            selected_label = st.selectbox(
                "选择已有课程",
                list(course_labels.keys()),
                key="sample_existing",
            )
            target_course_id = course_labels[selected_label]
            st.caption(f"目标目录：`{registry.course_dir(target_course_id)}`")

    can_import = source_ok and len(source_files) > 0 and (
        (target_mode == "新建课程后导入" and bool(new_course_name))
        or (target_mode == "导入到已有课程" and target_course_id is not None)
    )

    if st.button("一键导入样例课件", type="primary", disabled=not can_import):
        status_box = st.empty()
        progress = st.progress(0.0, text="准备导入…")
        try:
            with st.spinner("正在导入课件，请稍候…"):
                registry = refresh_registry()
                ingest = get_ingest()

                if target_mode == "新建课程后导入":
                    name = new_course_name
                    existing_names = {c.name for c in registry.list_courses()}
                    if name in existing_names:
                        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
                        name = f"{name}-{suffix}"
                    status_box.info(f"正在创建课程「{name}」…")
                    course = registry.create_course(
                        name,
                        description=f"从目录导入：{source_dir}",
                    )
                    target_course_id = course.course_id
                else:
                    if not target_course_id:
                        raise ValueError("请选择已有课程")
                    registry.get_course(target_course_id)

                dest_dir = registry.course_dir(target_course_id)
                results = []
                total = len(source_files)
                for i, path in enumerate(source_files):
                    status_box.info(f"正在处理（{i + 1}/{total}）：{path.name}")
                    progress.progress((i) / total, text=f"向量化 {path.name}")
                    results.append(ingest.ingest_file(target_course_id, path))
                    progress.progress((i + 1) / total, text=f"已完成 {i + 1}/{total}")

            refresh_registry()
            ok_n = sum(1 for r in results if r.ok)
            st.session_state["last_import_summary"] = (
                f"导入完成：成功 {ok_n} / {len(results)}；"
                f"课程 ID：`{target_course_id}`；目录：`{dest_dir}`"
            )
            st.session_state["last_import_lines"] = [
                f"- {r.filename} → {r.chunk_count} chunks" for r in results if r.ok
            ]
            st.session_state["last_import_errors"] = [
                f"- {r.filename} 失败：{r.error}" for r in results if not r.ok
            ]
            st.rerun()
        except Exception as exc:
            progress.empty()
            status_box.empty()
            st.error(str(exc))

with tab_manage:
    courses = refresh_registry().list_courses()
    if not courses:
        st.info("暂无课程。")
    else:
        course_labels = {f"{c.name} ({c.course_id})": c.course_id for c in courses}
        selected_label = st.selectbox("选择课程", list(course_labels.keys()), key="manage_course")
        course_id = course_labels[selected_label]
        course = registry.get_course(course_id)
        st.caption(f"课程目录：`{registry.course_dir(course_id)}`")

        if not course.documents:
            st.info("该课程暂无课件。")
        else:
            for doc in course.documents:
                cols = st.columns([4, 1, 1, 1])
                cols[0].write(
                    f"**{doc.filename}**  ·  {doc.chunk_count} chunks  ·  `{doc.doc_id[:8]}…`"
                )
                if cols[1].button("重新学习", key=f"re_{doc.doc_id}"):
                    try:
                        with st.spinner(f"重新学习 {doc.filename}…"):
                            result = ingest.reingest_document(course_id, doc.doc_id)
                        if result.ok:
                            st.success(f"已更新：{result.chunk_count} chunks")
                            refresh_registry()
                            st.rerun()
                        else:
                            st.error(result.error)
                    except Exception as exc:
                        st.error(str(exc))
                if cols[2].button("删除", key=f"rm_{doc.doc_id}"):
                    try:
                        ingest.delete_document(course_id, doc.doc_id)
                        st.success(f"已删除 {doc.filename}")
                        refresh_registry()
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
