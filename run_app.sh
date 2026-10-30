#!/usr/bin/env bash
# 启动 Streamlit 应用
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/streamlit run app.py
