# MAS 多智能体（Chainlit）

在聊天框输入任务描述即可运行主图（规划、RAG、代码执行、审核、汇总）。

## 流式输出说明

- **Chainlit 支持流式**：`Message.stream_token`、带流式能力的 `LangchainCallbackHandler` 等，见 [Chainlit Streaming](https://docs.chainlit.io/advanced-features/streaming)。
- **本应用**（`app.py`）：
  - 使用 LangGraph 的 **`astream_events`**：每个顶层节点（Supervisor、RAG、Code Dev 等）一结束就更新一条 **Step**，不必等整条图跑完。
  - 对 **Finalize** 汇总节点：在 LLM 开启 `streaming` 时，将 **最终答案的 token** 流式写入聊天区；若事件不兼容或失败，会回退为「按节点增量 Step」或整段展示。
- **其他子图**（Supervisor / Code Dev 等）仍为 `invoke` 聚合输出；若要对某一步也做 token 级流式，需在对应 agent 内改为 `stream`/`astream` 并接到 Chainlit。

## 代码执行日志（压缩与开关）

- 默认在 Step 中展示 **压缩执行日志**（折叠 pip 等噪声，保留 Traceback / `===RESULT===` 等）。
- 环境变量（可选）：
  - `MAS_CHAINLIT_SHOW_FULL_EXEC_LOG=1`：在 Step 中额外展示完整容器 stdout。
  - `MAS_SAVE_FULL_EXEC_LOG=1`：将完整日志写入 `results/exec_full_*.log` 并在 Step 中提示路径。
  - `MAS_CHAINLIT_RESULT_JSON_FULL_OUTPUT=1`：`chainlit_result_*.json` 中的 `code_solution` 保留完整 `output` 字段（默认省略以减小文件）。
- Finalize 汇总 LLM 默认只接收 **压缩日志**；若需把完整 stdout 喂给 Finalize，设置 `MAS_FULL_EXEC_LOG_IN_FINALIZE=1`。
- **`MAS_CHAINLIT_IMAGE_THUMBS=1`**：对大图生成 JPEG 缩略图再发给 Chainlit（需 Pillow）；**默认关闭**，避免首次打开大图时的解码/写盘延迟。

## 链接

- [Chainlit 文档](https://docs.chainlit.io)
- [LangGraph 流式](https://docs.langchain.com/oss/python/langgraph/streaming)
