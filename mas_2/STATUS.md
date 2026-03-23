# MAS-2 功能检查与改进说明

本文档记录当前各项功能的检查结果、已修复问题与后续建议。

---

## 一、已修复问题（本次检查）

### 1. Tool Caller 从未被正确路由（严重）

**现象**：Supervisor 决定派发 `tool_caller` 时，主图没有进入 Tool Caller 节点，而是直接进入 finalize，导致「很少会调用工具」。

**原因**：`src/main.py` 的 `supervisor_router` 只处理了 `next_worker == "data_analyst"` 并映射到节点 `"tool_caller"`，而 Supervisor 实际返回的是 `next_worker == "tool_caller"`。未显式处理 `"tool_caller"` 时，会落入 `else` 分支返回 `"finalize"`。

**修复**：
- 在 `supervisor_router` 中同时接受 `next_worker in ("data_analyst", "tool_caller")`，均路由到节点 `"tool_caller"`。
- 在 Critic 中，审核通过后写入 `final_report` 时，同时认可 `last_worker == "tool_caller"`（与 `data_analyst` 一致）。

### 2. RAG 无文档入库，检索始终为空

**现象**：RAG 依赖 Chroma 向量库，但项目内没有任何逻辑向 Chroma 写入文档，导致检索结果始终为「未找到相关文献片段」或「向量库未配置」。

**修复**：
- 在 `src/agents/rag_researcher/graph.py` 中新增 `add_documents_to_collection(documents, ids, metadatas)`，用于将文档写入当前配置的 Chroma 集合。
- 新增脚本 `scripts/ingest_rag.py`：对指定目录下的 `.txt`/`.md` 文件做入库，使用方式见下文「RAG 使用前准备」。

---

## 二、RAG 使用前准备

1. 安装依赖：`pip install chromadb sentence-transformers`
2. 准备文档目录，例如 `./docs`，放入 `.txt` 或 `.md` 文件。
3. 执行入库（与主项目同目录下）：

   ```bash
   cd mas_2
   python scripts/ingest_rag.py --dir ./docs
   ```

4. 可选环境变量（与 `rag_researcher/graph.py` 一致）：
   - `CHROMA_PERSIST_PATH`：Chroma 持久化根目录；未设置时默认为 **`<mas_2 根>/chroma_db`**（与 CWD 无关，见 `src/utils/project_paths.py`）。自定义时建议绝对路径，并与入库脚本一致。
   - `CHROMA_COLLECTION`：集合名（默认 `default_collection`）

未执行入库前，RAG Researcher 仍会运行，但检索结果为空，效果不符合预期属正常现象。

### Workflow 技能（试点）

- **注册与计划**：[`src/utils/workflow_skills.py`](src/utils/workflow_skills.py) 扫描 `workflows/*/SKILL.md`；Supervisor 生成计划时注入技能短目录，每步可填可选字段 **`skill_id`**（与 `id=\`...\`` 一致）。
- **状态**：`GlobalState.current_step_skill_id` 随当前步骤同步；Finalize 展示计划时可带 `skill_id`。
- **Code Dev**：试点三条（`gwas-to-function-twas`、`scrna-trajectory-inference`、`scrnaseq-scanpy-core-analysis`）在 Docker 内将对应目录**只读挂载**为 `/app/workflow`，并设置 `PYTHONPATH`；仅 **`scrnaseq-scanpy-core-analysis`** 沿用原 Scanpy 专用 system prompt 与执行头；其余两条使用通用 workflow 提示词与非 Scanpy 执行头。
- **Critic**：审核代码时附带当前 `skill_id` 说明，避免用不相关的 Scanpy 标准误杀。
- **向量入库（可选）**：对试点目录批量入库可运行：

  ```bash
  cd mas_2
  python scripts/ingest_workflow_pilots.py
  ```

  （内部多次调用 `scripts/parse_docs.py`，与 `CHROMA_PERSIST_PATH` / `--target-collection` 等参数兼容。）

---

## 三、当前功能状态概览

| 模块 | 状态 | 说明 |
|------|------|------|
| Supervisor | ✅ | 计划生成 + 按步骤/动态派遣；已能正确派发 tool_caller |
| Code Dev | ✅ | 代码生成与执行；支持试点 workflow 挂载 `/app/workflow` 与按 skill 切换提示词 |
| Critic | ✅ | 代码/文档/工具结果审核；通过后正确写入 code_solution / rag_context / final_report |
| RAG Researcher | ⚠️ 需先入库 | 检索逻辑正常；**需先运行 `scripts/ingest_rag.py` 写入文档** |
| Tool Caller | ✅ | 路由已修复；工具注册与执行正常，需 Supervisor 派发到本节点 |
| Finalize | ✅ | 汇总 rag_context、code_solution、final_report 输出 |

---

## 四、为何「工具很少被调用」——逻辑说明

即使路由已修复，Tool Caller 的触发仍依赖 Supervisor 的**派发决策**：

1. **有计划时**：根据当前步骤的 `name`/`description` 做关键词匹配，只有包含「工具、数据库、调用、tool、database、查询基因、鉴定」等才会派到 `tool_caller`。若 LLM 生成的计划里步骤描述不用这些词，就不会派发。
2. **无计划或动态决策时**：由 LLM 在 `rag_researcher / code_dev / tool_caller / critic / FINISH` 中选一个。Prompt 中已说明「如果需要调用现有工具，调用 tool_caller」，但模型可能仍偏好 code_dev 或直接 FINISH。

**建议**：
- 在计划生成的 system prompt 中明确列举「需要调用生物信息工具（如基因查询、富集、细胞类型注释）时，步骤描述中应包含“调用工具”或“使用 Tool Caller”」。
- 在动态决策的 user prompt 中，当用户问题明显涉及「查基因、富集、细胞类型」时，可增加一句示例：例如「用户要求根据 DE 基因鉴定细胞类型 → 应派发 tool_caller」。

---

## 五、后续可选改进

- **RAG**：支持更多格式（如 PDF）、分块策略与元数据过滤；或提供默认示例文档便于开箱即用。
- **Tool Caller**：在 Supervisor 计划阶段或首次决策前，将当前已注册工具列表（名称+简要描述）注入 prompt，减少误判为「写代码」而非「调工具」。
- **可观测性**：在 Chainlit 或日志中打印每步的 `next_worker` 与 `last_worker`，便于确认是否进入 tool_caller 与 critic 的归档逻辑。
