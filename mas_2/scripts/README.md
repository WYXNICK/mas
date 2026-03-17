# parse_docs.py 使用说明

该脚本用于将指定目录中的 `txt / md / pdf` 文档解析、切分、向量化并写入 ChromaDB。

- PDF 通过 MinerU 解析为 Markdown。
- 默认在 `tmp_minerU_md` 目录仅保留最终 Markdown 文件（中间产物会清理）。
- 支持增量入库（仅处理新增/变化文件）。
- 支持全量重建。
- 支持 GPU 优先并在失败时回退 CPU。
- 默认写入 `default_collection`，也可通过参数指定本次写入集合。
- 向量库存储目录按集合名分目录，例如：`chroma_db/default_collection/`。
- 状态文件会记录每个文件的 `vectorized_at`（成功向量化时间）。

## 常用命令

1. 增量入库（默认，仅处理新增/变化文件）

```bash
python scripts/parse_docs.py --docs-dir ./rag_docs --ingest-mode incremental --run-recall-test --query "请总结文档核心内容" --top-k 3
```

2. 增量入库 + 删除目录中已不存在文件的旧向量

```bash
python scripts/parse_docs.py --docs-dir ./rag_docs --ingest-mode incremental --remove-missing-files --run-recall-test
```

3. 全量重建（重处理全部文件并清空集合）

```bash
python scripts/parse_docs.py --docs-dir ./rag_docs --ingest-mode all --reset-collection --run-recall-test
```

说明：`--reset-collection` 会清空集合内容并复用同一集合目录，不会删除并重建集合。

4. PDF 解析加速：GPU 优先，失败自动回退 CPU

```bash
python scripts/parse_docs.py --docs-dir ./rag_docs --mineru-backend pipeline --mineru-device auto --cuda-retries 1 --fallback-cpu --ingest-mode incremental --run-recall-test
```

5. 扫描版 PDF（OCR 模式）

```bash
python scripts/parse_docs.py --docs-dir ./rag_docs --mineru-method ocr --mineru-backend pipeline --mineru-device auto --ingest-mode incremental --run-recall-test
```

6. 自定义路径（文档目录、Chroma 路径、状态文件）

```bash
python scripts/parse_docs.py --docs-dir ./my_docs --chroma-path ./chroma_db --state-path ./chroma_db/default_collection/ingest_state.json --target-collection default_collection
```

7. 指定本次解析保存到新集合

```bash
python scripts/parse_docs.py --docs-dir ./rag_docs --target-collection my_custom_collection --run-recall-test
```

## 关键参数

- `--docs-dir`：文档目录（递归扫描 txt/md/pdf）
- `--ingest-mode`：`incremental` 或 `all`
- `--remove-missing-files`：删除已从目录移除文件对应的向量
- `--reset-collection`：入库前清空 collection
- `--target-collection`：本次写入集合名（默认 `default_collection`）
- `--mineru-backend`：MinerU 后端（默认 `pipeline`）
- `--mineru-device`：`auto/cpu/cuda/cuda:0` 等（仅 pipeline 生效）
- `--cuda-retries`：GPU 模式失败时的额外重试次数（默认 1）
- `--fallback-cpu`：GPU 失败自动回退 CPU（默认开启）
- `--chunk-size` / `--chunk-overlap`：切分策略
- `--run-recall-test`：入库后执行简单召回验证

## GPU 说明

- 如果日志提示“检测到当前 PyTorch 不支持 CUDA，将使用 CPU”，说明当前环境中的 PyTorch 是 CPU 版本。
- 这会导致 MinerU 的 `pipeline/cuda` 无法正常工作（常见报错为 `Torch not compiled with CUDA enabled`）。
- 解决方案：安装与本机 CUDA/驱动匹配的 CUDA 版 PyTorch，或继续使用 CPU 模式。
- 若在 Windows 上遇到 `libiomp5md.dll already initialized`，脚本已内置 `KMP_DUPLICATE_LIB_OK=TRUE` 兜底以避免进程中断。
