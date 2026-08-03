"""课程管理：新建 / 重命名 / 删除。"""
from __future__ import annotations

import streamlit as st

from src.config import ensure_data_dirs
from src.ui_state import get_registry, refresh_registry

ensure_data_dirs()

st.title("课程管理")
registry = get_registry()

with st.form("create_course"):
    st.subheader("新建课程")
    name = st.text_input("课程名称", placeholder="例如：组织绩效的战略领导力")
    description = st.text_area("课程简介", placeholder="可选")
    submitted = st.form_submit_button("创建课程")
    if submitted:
        try:
            course = registry.create_course(name, description)
            st.success(f"已创建课程：{course.name}（{course.course_id}）")
            refresh_registry()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.subheader("已有课程")
courses = registry.list_courses()
if not courses:
    st.info("暂无课程，请先新建。")
else:
    for course in courses:
        with st.expander(f"{course.name}  ·  {len(course.documents)} 份课件", expanded=False):
            st.write(f"ID：`{course.course_id}`")
            st.write(f"简介：{course.description or '（无）'}")
            st.write(f"创建时间：{course.created_at}")
            if course.documents:
                st.markdown("**课件列表**")
                for doc in course.documents:
                    st.write(f"- {doc.filename}（{doc.chunk_count} chunks）")

            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("新名称", value=course.name, key=f"rename_{course.course_id}")
                new_desc = st.text_input(
                    "新简介",
                    value=course.description,
                    key=f"desc_{course.course_id}",
                )
                if st.button("保存修改", key=f"save_{course.course_id}"):
                    try:
                        registry.rename_course(course.course_id, new_name, new_desc)
                        st.success("已更新")
                        refresh_registry()
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with col2:
                st.write("删除课程将同时删除该课全部向量库与原始课件，不可恢复。")
                confirm = st.checkbox("确认删除", key=f"confirm_{course.course_id}")
                if st.button("删除课程", key=f"del_{course.course_id}", disabled=not confirm):
                    try:
                        registry.delete_course(course.course_id)
                        st.success("已删除")
                        refresh_registry()
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
