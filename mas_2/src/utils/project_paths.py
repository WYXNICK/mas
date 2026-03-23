"""
mas_2 项目根目录与默认资源路径解析。

避免依赖进程 CWD：未设置环境变量时，默认路径锚定在仓库根（mas_2/）。
"""
from __future__ import annotations

import os
from pathlib import Path


def get_mas2_project_root() -> Path:
    """返回 mas_2 项目根目录（包含 src/、scripts/ 的目录）。"""
    # 本文件: mas_2/src/utils/project_paths.py -> parents[2] == mas_2
    root = Path(__file__).resolve().parents[2]
    marker_main = root / "src" / "main.py"
    marker_parse = root / "scripts" / "parse_docs.py"
    if not marker_main.is_file() or not marker_parse.is_file():
        raise RuntimeError(
            f"推断的项目根无效（缺少预期文件）: {root}\n"
            "请确认 mas_2 目录结构完整，或将 CHROMA_PERSIST_PATH 设为绝对路径。"
        )
    return root


def resolve_chroma_persist_path() -> Path:
    """
    Chroma 持久化根目录（不含集合子目录；与 parse_docs / rag_researcher 约定一致）。

    - 若设置 CHROMA_PERSIST_PATH：expanduser 后返回（相对路径仍相对当前 CWD，便于显式覆盖）。
    - 若未设置：<mas_2 根>/chroma_db
    """
    raw = os.getenv("CHROMA_PERSIST_PATH")
    if raw is not None and str(raw).strip():
        return Path(raw).expanduser()
    return get_mas2_project_root() / "chroma_db"
