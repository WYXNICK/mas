"""
使用 MinerU + ChromaDB 构建 RAG 文档库。

功能:
1. 扫描指定目录下的 txt/md/pdf 文件（递归）
2. pdf 使用 MinerU 解析为 markdown
3. 统一进行 chunk
4. 计算 embedding 并写入 ChromaDB
5. 支持全量或增量入库
6. 支持 MinerU GPU 优先，失败自动回退 CPU
7. 可选执行一次简单召回测试

示例:
    cd mas_2
    python scripts/parse_docs.py --docs-dir ./rag_docs --run-recall-test
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows + CUDA 环境下，部分库可能重复加载 OpenMP 运行时导致 torch 导入失败。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

DEFAULT_DOCS_DIR = _PROJECT_DIR / "rag_docs"
DEFAULT_CHROMA_PATH = Path(os.getenv("CHROMA_PERSIST_PATH", str(_PROJECT_DIR / "chroma_db")))
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "default_collection")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class _STEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, normalize_embeddings=True).tolist()

    def name(self):
        return "sentence-transformer"


def _read_text_file(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"无法解码文件: {path}")


def _find_best_markdown(output_dir: Path, source_pdf: Path) -> Optional[Path]:
    candidates = list(output_dir.rglob("*.md"))
    if not candidates:
        return None

    stem = source_pdf.stem.lower()
    stem_hits = [p for p in candidates if stem in p.stem.lower() or stem in p.name.lower()]
    pool = stem_hits if stem_hits else candidates
    return max(pool, key=lambda p: p.stat().st_size)


def _detect_device_for_pipeline(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch

        # 仅当 PyTorch 具备 CUDA 且可用时才走 GPU。
        if torch.version.cuda is not None and torch.cuda.is_available():
            return "cuda"

        print("    检测到当前 PyTorch 不支持 CUDA，将使用 CPU。")
    except Exception:
        print("    无法检测 PyTorch CUDA 能力，将使用 CPU。")

    return "cpu"


def _sanitize_collection_dir_name(name: str) -> str:
    invalid = '<>:"/\\|?*'
    sanitized = "".join("_" if ch in invalid else ch for ch in name).strip()
    return sanitized or "default_collection"


def _run_mineru_command(cmd: List[str]) -> Tuple[bool, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    merged_output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result.returncode == 0, merged_output


def _parse_pdf_with_mineru(
    pdf_path: Path,
    mineru_output_root: Path,
    method: str,
    backend: str,
    device: str,
    cuda_retries: int,
    fallback_cpu: bool,
) -> str:
    rel_key = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:10]
    out_dir = mineru_output_root / f"{pdf_path.stem}_{rel_key}"
    final_md_path = mineru_output_root / f"{pdf_path.stem}_{rel_key}.md"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "mineru",
        "-p",
        str(pdf_path),
        "-o",
        str(out_dir),
        "-m",
        method,
        "-b",
        backend,
    ]

    attempts: List[Tuple[str, List[str]]] = []
    if backend == "pipeline":
        selected_device = _detect_device_for_pipeline(device)
        attempts.append((f"pipeline/{selected_device}", base_cmd + ["-d", selected_device]))
        if selected_device.startswith("cuda") and cuda_retries > 0:
            for i in range(cuda_retries):
                attempts.append((f"pipeline/{selected_device}-retry-{i+1}", base_cmd + ["-d", selected_device]))
        if fallback_cpu and selected_device != "cpu":
            attempts.append(("pipeline/cpu-fallback", base_cmd + ["-d", "cpu"]))
    else:
        attempts.append((backend, base_cmd))

    last_log = ""
    best_md: Optional[Path] = None
    for label, cmd in attempts:
        # 每次尝试前清理目录，避免旧产物干扰本次判断。
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        ok, logs = _run_mineru_command(cmd)
        if not ok:
            last_log = f"尝试 {label} 失败\n{logs}"
            print(f"    MinerU 解析失败，准备重试: {label}")
            continue

        best_md = _find_best_markdown(out_dir, pdf_path)
        if best_md is not None:
            print(f"    MinerU 解析成功: {label}")
            break

        last_log = f"尝试 {label} 未产出 markdown\n{logs}"
        print(f"    MinerU 未产出 markdown，准备重试: {label}")
    else:
        raise RuntimeError(f"MinerU 解析失败\nPDF: {pdf_path}\n{last_log}")

    if best_md is None:
        raise RuntimeError(f"MinerU 未生成 markdown: {pdf_path}")

    md_text = _read_text_file(best_md)

    # 仅保留最终 md，其余中间产物删除。
    final_md_path.write_text(md_text, encoding="utf-8")
    shutil.rmtree(out_dir, ignore_errors=True)

    return md_text


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            breakpoints = [window.rfind("\n\n"), window.rfind("\n"), window.rfind("。"), window.rfind(". ")]
            best = max(breakpoints)
            if best > int(chunk_size * 0.6):
                end = start + best + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        next_start = max(0, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _iter_source_files(docs_dir: Path) -> List[Path]:
    files = [p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files)


def _build_doc_id(namespace: str, relative_path: str, chunk_index: int) -> str:
    raw = f"{namespace}::{relative_path}::{chunk_index}"
    return f"doc_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def _file_fingerprint(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(data).hexdigest()


def _load_state(state_path: Path) -> Dict:
    if not state_path.exists():
        return {"version": 1, "files": {}}

    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "files": {}}


def _save_state(state_path: Path, state: Dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _delete_ids(collection, ids: List[str]) -> None:
    if not ids:
        return
    unique_ids = sorted(set(ids))
    collection.delete(ids=unique_ids)


def _ingest(args: argparse.Namespace) -> None:
    docs_dir = Path(args.docs_dir).resolve()
    chroma_root_path = Path(args.chroma_path).resolve()
    mineru_output_dir = Path(args.mineru_output_dir).resolve()
    source_namespace = hashlib.sha1(str(docs_dir).encode("utf-8")).hexdigest()[:12]
    collection_name = args.target_collection or args.collection or DEFAULT_COLLECTION
    collection_dir_name = _sanitize_collection_dir_name(collection_name)
    chroma_path = chroma_root_path / collection_dir_name
    if args.state_path:
        state_path = Path(args.state_path).resolve()
    else:
        state_path = chroma_path / "ingest_state.json"

    if not docs_dir.exists() or not docs_dir.is_dir():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    source_files = _iter_source_files(docs_dir)
    if not source_files:
        raise FileNotFoundError(f"目录中未找到 txt/md/pdf 文件: {docs_dir}")

    print(f"[1/4] 加载 embedding 模型: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)
    embedding_fn = _STEmbeddingFunction(embedder)

    print(f"[2/4] 连接 ChromaDB: {chroma_path}")
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    if args.reset_collection:
        existing_count = collection.count()
        if existing_count > 0:
            existing = collection.get(include=[], limit=existing_count)
            ids = existing.get("ids") or []
            if ids:
                collection.delete(ids=ids)
            print(f"已清空旧集合内容: {collection_name} ({existing_count} 条)")

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metadatas: List[dict] = []

    state = _load_state(state_path)
    old_files: Dict = state.get("files", {})
    new_files: Dict = {}

    pending_delete_ids: List[str] = []
    skipped_unchanged = 0
    processed_count = 0
    failed_count = 0

    mineru_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] 处理文档，共 {len(source_files)} 个文件，模式={args.ingest_mode}")
    current_rel_paths = set()

    for idx, path in enumerate(source_files, start=1):
        suffix = path.suffix.lower()
        rel_path = path.relative_to(docs_dir).as_posix()
        current_rel_paths.add(rel_path)

        fp = _file_fingerprint(path)
        old_entry = old_files.get(rel_path)
        unchanged = old_entry is not None and old_entry.get("fingerprint") == fp

        if args.ingest_mode == "incremental" and unchanged:
            new_files[rel_path] = old_entry
            skipped_unchanged += 1
            print(f"  - ({idx}/{len(source_files)}) {rel_path} [跳过: 未变化]")
            continue

        print(f"  - ({idx}/{len(source_files)}) {rel_path}")

        try:
            if suffix == ".pdf":
                text = _parse_pdf_with_mineru(
                    path,
                    mineru_output_dir,
                    args.mineru_method,
                    args.mineru_backend,
                    args.mineru_device,
                    args.cuda_retries,
                    args.fallback_cpu,
                )
            else:
                text = _read_text_file(path)
        except Exception as exc:
            print(f"    跳过（解析失败）: {exc}")
            failed_count += 1
            if old_entry:
                new_files[rel_path] = old_entry
            continue

        chunks = _chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        if not chunks:
            print("    跳过（无有效文本）")
            failed_count += 1
            if old_entry:
                new_files[rel_path] = old_entry
            continue

        if old_entry and old_entry.get("chunk_ids"):
            pending_delete_ids.extend(old_entry.get("chunk_ids", []))

        current_chunk_ids: List[str] = []

        for cidx, chunk in enumerate(chunks):
            doc_id = _build_doc_id(source_namespace, rel_path, cidx)
            all_ids.append(doc_id)
            current_chunk_ids.append(doc_id)
            all_docs.append(chunk)
            all_metadatas.append(
                {
                    "source": rel_path,
                    "file_type": suffix.lstrip("."),
                    "chunk_index": cidx,
                    "total_chunks": len(chunks),
                }
            )

        new_files[rel_path] = {
            "fingerprint": fp,
            "chunk_ids": current_chunk_ids,
            "file_type": suffix.lstrip("."),
            "vectorized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        processed_count += 1

    if args.remove_missing_files:
        missing_paths = sorted(set(old_files.keys()) - current_rel_paths)
        for missing_rel_path in missing_paths:
            old_entry = old_files.get(missing_rel_path, {})
            pending_delete_ids.extend(old_entry.get("chunk_ids", []))
            print(f"  - 删除缺失文件向量: {missing_rel_path}")

    if pending_delete_ids:
        print(f"[4/5] 清理旧向量: {len(set(pending_delete_ids))} 条")
        _delete_ids(collection, pending_delete_ids)

    if all_docs:
        print(f"[5/5] 写入向量库: {len(all_docs)} 个 chunks")
        collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_metadatas)
    else:
        print("[5/5] 无需写入新向量（全部跳过或解析失败）")

    if not all_docs and not pending_delete_ids and processed_count == 0 and skipped_unchanged == 0:
        raise RuntimeError("没有可入库的 chunk，请检查文档内容或解析日志")

    print(f"写入完成，集合 '{collection_name}' 当前总文档数: {collection.count()}")
    print(
        f"统计: 处理={processed_count}, 跳过未变={skipped_unchanged}, "
        f"失败={failed_count}, 删除旧向量={len(set(pending_delete_ids))}"
    )

    new_state = {
        "version": 1,
        "collection": collection_name,
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "files": new_files,
    }
    _save_state(state_path, new_state)
    print(f"状态文件已更新: {state_path}")

    if args.run_recall_test:
        print("\n--- 简单召回测试 ---")
        query = args.query.strip() if args.query else "请总结文档的核心主题"
        results = collection.query(query_texts=[query], n_results=args.top_k)
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        if not docs:
            print("未召回到结果")
            return

        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            source = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
            preview = doc[:120].replace("\n", " ")
            print(f"[{i}] source={source} | {preview}...")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析 txt/md/pdf 并写入 ChromaDB")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="文档目录（与 chroma_db 并列）")
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA_PATH), help="ChromaDB 持久化目录")
    parser.add_argument("--target-collection", default=None, help=f"本次写入的 collection 名称（默认: {DEFAULT_COLLECTION}）")
    parser.add_argument("--collection", default=None, help="兼容旧参数；建议改用 --target-collection")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding 模型")
    parser.add_argument("--state-path", default=None, help="增量状态文件路径（默认: chroma_path/collection/ingest_state.json）")
    parser.add_argument(
        "--ingest-mode",
        choices=["all", "incremental"],
        default="incremental",
        help="all=全量重处理全部文件, incremental=仅处理新增/变化文件",
    )
    parser.add_argument(
        "--remove-missing-files",
        action="store_true",
        help="从向量库中删除已不存在于 docs-dir 的文件对应向量",
    )
    parser.add_argument("--chunk-size", type=int, default=2000, help="chunk 大小（字符数）")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="chunk overlap（字符数）")
    parser.add_argument("--mineru-output-dir", default=str(_PROJECT_DIR / "tmp_minerU_md"), help="MinerU markdown 输出目录（仅保留最终 md）")
    parser.add_argument("--mineru-method", choices=["auto", "txt", "ocr"], default="auto", help="MinerU PDF 解析模式")
    parser.add_argument(
        "--mineru-backend",
        choices=["pipeline", "vlm-http-client", "hybrid-http-client", "vlm-auto-engine", "hybrid-auto-engine"],
        default="pipeline",
        help="MinerU 后端，pipeline 通常更快更稳",
    )
    parser.add_argument(
        "--mineru-device",
        default="auto",
        help="仅 pipeline 后端生效。auto/cpu/cuda/cuda:0 等，auto 会优先 CUDA 否则 CPU",
    )
    parser.add_argument(
        "--cuda-retries",
        type=int,
        default=1,
        help="当 pipeline/cuda 失败或未产出 markdown 时的额外重试次数",
    )
    parser.add_argument(
        "--fallback-cpu",
        dest="fallback_cpu",
        action="store_true",
        help="当 pipeline 使用 GPU 失败时自动回退 CPU 重试（默认开启）",
    )
    parser.add_argument(
        "--no-fallback-cpu",
        dest="fallback_cpu",
        action="store_false",
        help="禁用 GPU 失败后的 CPU 回退",
    )
    parser.set_defaults(fallback_cpu=True)
    parser.add_argument("--reset-collection", action="store_true", help="入库前清空 collection")
    parser.add_argument("--run-recall-test", action="store_true", help="入库后执行召回测试")
    parser.add_argument("--query", default="", help="召回测试使用的查询语句")
    parser.add_argument("--top-k", type=int, default=3, help="召回测试返回条数")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap 必须小于 --chunk-size")

    _ingest(args)


if __name__ == "__main__":
    main()
