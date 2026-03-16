# MAS-2：单细胞分析多智能体系统

基于 **LangGraph** 和 **Chainlit** 构建的多智能体系统，能够根据用户的自然语言指令，自动规划、编写代码、执行分析并汇总报告，专注于单细胞 RNA 测序（scRNA-seq）数据分析场景。

---

## 目录结构

```
mas_2/
├── app.py                        # Chainlit 前端入口
├── chainlit.md                   # 聊天界面欢迎语
├── requirements.txt              # Python 依赖
├── langgraph.json                # LangGraph Studio 配置
├── .env / .env.example           # 环境变量
├── .chainlit/
│   └── config.toml               # Chainlit 服务配置
├── chroma_db/                    # ChromaDB 向量数据库（持久化）
├── data/                         # 输入数据（如 10x Genomics hg19 矩阵）
├── notebooks/
│   └── test_single_cell_analysis.ipynb  # 流程测试 Notebook
├── results/                      # 执行结果（JSON、图片等）
├── src/
│   ├── main.py                   # 主图编排（LangGraph 主入口）
│   ├── core/
│   │   ├── config.py             # 全局配置（模型、API Key 等）
│   │   ├── state.py              # 全局状态定义（GlobalState）
│   │   └── llm.py                # LLM 工厂函数
│   └── agents/
│       ├── supervisor/           # 调度智能体
│       ├── code_dev/             # 代码开发与执行智能体
│       ├── critic/               # 代码与结果审核智能体
│       ├── rag_researcher/       # 文献检索智能体
│       └── tool_caller/          # 生物信息工具调用智能体
└── tests/
    ├── unit/
    └── integration/
```

---

## 系统架构

### 整体流程

```
用户输入
   │
   ▼
Supervisor（规划与调度）
   │ 根据计划步骤决定下一个 Worker
   ├──► RAG Researcher（文献/知识检索）
   ├──► Code Dev（代码生成与执行）
   ├──► Tool Caller（生物信息工具调用）
   │
   └──► Critic（产出审核）
           │
           ├── 通过 ──► 返回 Supervisor（继续下一步）
           └── 不通过 ──► 返回对应 Worker（重试）
                                    │
                              Supervisor 判断全部完成
                                    │
                                    ▼
                               Finalize（汇总报告）
```

### 状态流转（GlobalState）

所有 Agent 共享同一个 `GlobalState`，关键字段：

| 字段 | 说明 |
|------|------|
| `messages` | 对话历史 |
| `user_query` | 用户原始问题 |
| `plan` | 执行计划（PlanStep 列表） |
| `current_step_index` | 当前执行步骤序号 |
| `next_worker` | 下一步派遣的 Worker |
| `pending_contribution` | Worker 产出，等待 Critic 审核 |
| `critique_feedback` | Critic 反馈（不通过时） |
| `is_approved` | 当前步骤是否通过审核 |
| `rag_context` | RAG 检索到的知识 |
| `final_answer` | 最终汇总答案 |

---

## 各智能体说明

### Supervisor（`src/agents/supervisor/graph.py`）

**职责**：解析用户需求，生成分步执行计划，并在每轮循环后决定下一个执行的 Worker。

**计划步骤（PlanStep）包含**：
- `step_id`、`name`、`description`
- `input_files`、`output_files`（文件路径规划）
- `acceptance_criteria`（验收标准，传递给 Critic）

**调度逻辑**：
- 步骤描述含"代码/编程/执行/分析" → 派遣 `code_dev`
- 步骤描述含"检索/文献/知识" → 派遣 `rag_researcher`
- 步骤描述含"工具/数据库/查询基因/鉴定" → 派遣 `tool_caller`
- 全部步骤完成 → `FINISH`（触发 Finalize）

---

### Code Dev（`src/agents/code_dev/graph.py`）

**职责**：根据当前步骤描述，生成并在 Docker 沙箱中执行 Python 代码（主要为 Scanpy 单细胞分析代码）。

**工作流**：

```
generate_code → self_reflection → execute_code
                                      │
                               成功？─┤
                               否     ▼ 是
                         prepare_retry    display_result → END
                               │
                         (最多重试 3 次)
```

**关键实现**：
- **代码生成**：LLM 生成 Python 脚本及对应 `requirements.txt`
- **安全检查**：`self_reflection` 过滤 `eval`、`exec`、`open` 等危险调用
- **Docker 执行**：`CodeExecutor` 启动容器，挂载 `/app/data`（数据目录）和 `/app/output`（输出目录）
- **图片处理**：执行产出的 PNG 图片经 base64 编码后回传，支持在前端展示

---

### Critic（`src/agents/critic/graph.py`）

**职责**：审核每个 Worker 的产出，判断是否满足当前步骤的验收标准。

**审核策略**：

| 产出类型 | 审核方式 |
|----------|----------|
| 图片（UMAP 等） | 调用视觉模型（`qwen-vl-plus`）分析图像质量 |
| 代码/执行结果 | LLM 对照验收标准、执行日志进行评判 |
| 文献检索结果 | LLM 评估相关性与完整性 |
| 工具调用结果 | LLM 评估结果有效性 |

**审核规则**（内置 Prompt）：
- 只审核当前步骤，不要求产出完成整个分析
- Docker 路径映射差异（`/app/data` vs 本地路径）视为正常
- 执行日志优先：若日志显示成功，即使代码看起来有问题也通过

---

### RAG Researcher（`src/agents/rag_researcher/graph.py`）

**职责**：从本地向量数据库中检索与当前步骤相关的文献知识。

**实现**：
- 向量库：ChromaDB（持久化到 `chroma_db/`）
- 嵌入模型：SentenceTransformer
- 检索结果写入 `retrieved_docs` 和 `pending_contribution`

---

### Tool Caller（`src/agents/tool_caller/graph.py`）

**职责**：调用生物信息学工具，并用 LLM 解释结果。

**工作流**：

```
decision（决定调用哪个工具）→ execute_tool（执行）→ interpret（解释结果）→ END
```

**内置工具**：

| 工具名 | 功能 |
|--------|------|
| `run_celltype_annotation` | 细胞类型注释（Enrichr + LLM 专家推断） |
| `gene_set_enrichment` | 基因集富集分析（gseapy GSEA/ORA） |
| `query_mygene` | 基因信息查询（MyGene.info API） |

---

## 前端（Chainlit）

### 位置

`mas_2/app.py`

### 启动方式

```bash
cd mas_2
chainlit run app.py -w
```

默认访问地址：`http://localhost:8000`

### 核心流程

1. 用户在聊天框输入分析需求（如"请对提供的 10x 数据进行聚类和细胞类型鉴定"）
2. `build_initial_state(user_query)` 构建初始状态
3. `graph.stream()` 流式运行主图，实时展示各 Agent 输出：
   - Supervisor：输出执行计划和当前步骤
   - Code Dev：展示生成的代码、执行日志、输出图片
   - Tool Caller：展示工具调用报告
   - Critic：展示审核结论与反馈
   - Finalize：展示最终汇总报告
4. 结果保存到 `results/chainlit_result_<timestamp>.json`

### 配置文件

`.chainlit/config.toml` 控制会话超时、文件上传、MCP 集成等参数。

---

## 后端（LangGraph）

### 主图入口

`src/main.py:graph`

LangGraph Studio 配置（`langgraph.json`）：

```json
{
  "graphs": {
    "agent": "src/main.py:graph"
  }
}
```

### 直接调用（不经过 Chainlit）

```python
from src.main import graph
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [HumanMessage(content="分析 data/ 目录下的单细胞数据")],
    "user_query": "分析 data/ 目录下的单细胞数据",
    "plan": [],
    "current_step_index": 0,
    "next_worker": None,
    "last_worker": None,
    "pending_contribution": None,
    "critique_feedback": None,
    "is_approved": False,
}

config = {"configurable": {"thread_id": "test-001"}}
result = graph.invoke(initial_state, config=config)
print(result["final_answer"])
```

### 使用 LangGraph Studio

```bash
cd mas_2
langgraph up
```

---

## 环境配置

### 依赖安装

```bash
cd mas_2
pip install -r requirements.txt
```

### 环境变量（`.env`）

```env
# LLM 服务（DashScope 兼容 OpenAI 接口）
API_KEY=your_dashscope_api_key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-turbo
TEMPERATURE=0.5

# LangSmith 追踪（可选）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=mas_2
```

### Docker 环境（代码执行沙箱）

Code Dev Agent 依赖 Docker 运行代码沙箱，请确保本机已安装并启动 Docker：

```bash
docker info  # 验证 Docker 可用
```

---

## 快速开始

```bash
# 1. 安装依赖
cd mas_2
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key

# 3. 准备数据
# 将 10x Genomics 数据放入 data/ 目录

# 4. 启动前端
chainlit run app.py -w

# 5. 浏览器访问 http://localhost:8000，输入分析需求
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 多智能体框架 | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | 阿里云百炼（Qwen-Turbo / Qwen-VL-Plus），兼容 OpenAI 接口 |
| 前端 | [Chainlit](https://github.com/Chainlit/chainlit) |
| 向量数据库 | [ChromaDB](https://www.trychroma.com/) + SentenceTransformer |
| 代码执行沙箱 | Docker |
| 生物信息库 | Scanpy, gseapy, MyGene.info API |
| 追踪监控 | LangSmith |
