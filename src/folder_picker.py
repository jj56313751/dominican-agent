"""本机目录选择（供本地 Streamlit 使用）。

macOS 上禁止在 Streamlit 脚本线程里调用 tkinter（会触发
NSWindow should only be instantiated on the main thread 并崩溃），
因此使用独立进程 osascript 弹出系统文件夹对话框。
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import List, Optional


def list_subdirectories(root: Path) -> List[Path]:
    """列出根目录下的一级子目录。"""
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def pick_directory(initial_dir: Optional[str] = None) -> Optional[str]:
    """弹出系统文件夹选择对话框，返回所选路径；取消则返回 None。"""
    system = platform.system()
    if system == "Darwin":
        return _pick_directory_macos(initial_dir)
    if system == "Windows":
        return _pick_directory_windows(initial_dir)
    raise RuntimeError(
        f"当前系统 {system} 不支持图形化选目录，请改用「快捷选择」下拉框。"
    )


def _pick_directory_macos(initial_dir: Optional[str] = None) -> Optional[str]:
    """通过 osascript 在独立进程中打开选文件夹对话框。"""
    lines = [
        'set theFolder to choose folder with prompt "选择数据源目录"',
    ]
    # default location 在部分系统上可用
    if initial_dir:
        init = Path(initial_dir).expanduser()
        if not init.exists():
            init = init.parent if init.parent.exists() else None
        elif init.is_file():
            init = init.parent
        if init and init.exists():
            # AppleScript 路径用冒号分隔的 HFS 路径更稳，这里用 POSIX file
            posix = str(init).replace("\\", "\\\\").replace('"', '\\"')
            lines = [
                f'set defaultFolder to POSIX file "{posix}"',
                'set theFolder to choose folder with prompt "选择数据源目录" default location defaultFolder',
            ]
    lines.append("return POSIX path of theFolder")
    script = "\n".join(lines)
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # 用户取消也会非 0
        return None
    path = (result.stdout or "").strip()
    if not path:
        return None
    # osascript 返回的目录路径常带尾部 /
    return str(Path(path))


def _pick_directory_windows(initial_dir: Optional[str] = None) -> Optional[str]:
    """Windows：用 PowerShell FolderBrowserDialog（独立进程，避免线程问题）。"""
    init = ""
    if initial_dir and Path(initial_dir).exists():
        init = str(Path(initial_dir))
    ps = f"""
Add-Type -AssemblyName System.Windows.Forms
$browser = New-Object System.Windows.Forms.FolderBrowserDialog
$browser.Description = '选择数据源目录'
$browser.SelectedPath = '{init.replace("'", "''")}'
if ($browser.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $browser.SelectedPath
}}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    path = (result.stdout or "").strip()
    return path or None
