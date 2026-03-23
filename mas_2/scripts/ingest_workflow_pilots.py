#!/usr/bin/env python3
"""
将 mas_2/workflows 下技能目录入库到 Chroma（复用 parse_docs）。

默认：扫描所有含 SKILL.md 的子目录，并为每条向量写入 metadata skill_id（与 RAG 过滤配合）。

用法（在 mas_2 目录下）:
  python scripts/ingest_workflow_pilots.py
  python scripts/ingest_workflow_pilots.py --pilot-only
  python scripts/ingest_workflow_pilots.py --skills cell-cell-communication scrnaseq-scanpy-core-analysis

与 rag_researcher 一致的环境变量:
  CHROMA_PERSIST_PATH, CHROMA_COLLECTION
其余参数可追加在命令末尾，传给 parse_docs（如 --ingest-mode all）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.utils.project_paths import resolve_chroma_persist_path  # noqa: E402

PILOT_SKILL_IDS = (
    "gwas-to-function-twas",
    "scrna-trajectory-inference",
    "scrnaseq-scanpy-core-analysis",
)


def _sanitize_collection_dir_name(name: str) -> str:
    invalid = '<>:"/\\|?*'
    sanitized = "".join("_" if ch in invalid else ch for ch in name).strip()
    return sanitized or "default_collection"


def _workflow_skill_dirs(root: Path) -> list[Path]:
    wf = root / "workflows"
    if not wf.is_dir():
        return []
    out: list[Path] = []
    for d in sorted(wf.iterdir()):
        if d.is_dir() and (d / "SKILL.md").is_file():
            out.append(d)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Workflow 技能文档批量入库 Chroma")
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help=f"仅入库试点三条: {', '.join(PILOT_SKILL_IDS)}",
    )
    parser.add_argument(
        "--skills",
        nargs="*",
        default=None,
        metavar="SKILL_ID",
        help="仅入库指定技能目录名（可多个）；不传则全部",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(resolve_chroma_persist_path()),
        help="Chroma 持久化根目录（与 parse_docs --chroma-path 一致）",
    )
    parser.add_argument(
        "--target-collection",
        default=os.environ.get("CHROMA_COLLECTION", "default_collection"),
        help="集合名（与 parse_docs --target-collection 一致）",
    )
    args, passthrough = parser.parse_known_args()

    os.chdir(_PROJECT_DIR)
    parse_docs = _PROJECT_DIR / "scripts" / "parse_docs.py"
    collection_dir_name = _sanitize_collection_dir_name(args.target_collection)
    chroma_sub = Path(args.chroma_path).resolve() / collection_dir_name
    chroma_sub.mkdir(parents=True, exist_ok=True)

    dirs = _workflow_skill_dirs(_PROJECT_DIR)
    if args.pilot_only:
        pilot_set = set(PILOT_SKILL_IDS)
        dirs = [d for d in dirs if d.name in pilot_set]
    if args.skills is not None and len(args.skills) > 0:
        allow = set(args.skills)
        dirs = [d for d in dirs if d.name in allow]

    if not dirs:
        print("[ingest_workflow] 没有匹配的 workflow 目录（需 workflows/<id>/SKILL.md）", file=sys.stderr)
        sys.exit(1)

    for d in dirs:
        skill_id = d.name
        state_path = chroma_sub / f"ingest_workflow_{skill_id}.json"
        cmd = [
            sys.executable,
            str(parse_docs),
            "--docs-dir",
            str(d),
            "--skill-id",
            skill_id,
            "--state-path",
            str(state_path),
            "--chroma-path",
            str(Path(args.chroma_path).resolve()),
            "--target-collection",
            args.target_collection,
            "--ingest-mode",
            "incremental",
        ] + passthrough
        print(f"[run] skill_id={skill_id}\n      {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
