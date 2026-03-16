"""
Chainlit 前端：MAS 多智能体系统测试与展示
运行: cd mas_2 && chainlit run app.py -w
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# 可可视化为图片的扩展名（用于 Code Dev 输出文件）
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

# 确保 mas_2 在路径中，以便 from src.main import graph 可用
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

import chainlit as cl


def build_initial_state(user_query: str) -> dict:
    """构建主图初始状态，与 notebook / 集成测试保持一致。"""
    q = (user_query or "").strip()
    return {
        "messages": [HumanMessage(content=q)],
        "user_query": q,
        "plan": [],
        "current_step_index": 0,
        "current_step_input": None,
        "current_step_expected_output": None,
        "current_step_file_paths": None,
        "next_worker": "rag_researcher",
        "last_worker": "",
        "final_report": "",
        "code_solution": "",
        "rag_context": "",
        "pending_contribution": None,
        "critique_feedback": None,
        "is_approved": False,
    }


def _run_graph_stream(initial_state: dict, config: dict) -> tuple[list[tuple[str, dict]], dict | None]:
    """执行 graph.stream，返回 (每步 (node_name, state) 列表, 最终 state)。"""
    app_dir = Path(__file__).resolve().parent
    for candidate in (app_dir, Path.cwd()):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from src.main import graph

    steps: list[tuple[str, dict]] = []
    final_state = None
    for step in graph.stream(initial_state, config=config):
        for node_name, state in step.items():
            steps.append((node_name, state))
            final_state = state
    return steps, final_state


def _code_dev_image_elements(output_files: list) -> list:
    """从 output_files 中筛选图片文件，构建 Chainlit Image 元素用于可视化。"""
    elements = []
    for f in output_files or []:
        path = f.get("path") or f.get("name")
        name = f.get("name") or (Path(path).name if path else "image")
        if not path or not isinstance(path, str):
            continue
        path = Path(path)
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        try:
            elements.append(
                cl.Image(path=str(path.resolve()), name=name, display="inline", size="medium")
            )
        except Exception:
            continue
    return elements


def _format_agent_output(node_name: str, state: dict) -> str:
    """从 state 中提取当前节点的可读输出，避免展示原始 JSON。"""
    parts: list[str] = []

    if node_name == "supervisor":
        plan = state.get("plan") or []
        if plan:
            lines = []
            for i, p in enumerate(plan, 1):
                name = p.get("name", "") if isinstance(p, dict) else getattr(p, "name", "")
                desc = p.get("description", "") if isinstance(p, dict) else getattr(p, "description", "")
                lines.append(f"{i}. **{name}**\n   {desc}")
            parts.append("**执行计划**\n" + "\n".join(lines))
        next_worker = state.get("next_worker", "")
        if next_worker:
            parts.append(f"\n下一步: **{next_worker}**")

    elif node_name == "rag_researcher":
        rag = state.get("rag_context", "")
        if rag:
            parts.append(rag if isinstance(rag, str) else str(rag))
        pending = state.get("pending_contribution")
        if pending and not rag:
            if isinstance(pending, list):
                parts.append("\n\n".join(str(x) for x in pending))
            else:
                parts.append(str(pending))

    elif node_name == "code_dev":
        code = state.get("code_solution", "")
        pending = state.get("pending_contribution")
        if isinstance(pending, dict):
            if pending.get("code"):
                parts.append("**代码**\n```python\n" + (pending["code"] or "") + "\n```")
            # 结果输出：与 result 内容分开，便于后续对结果文件做可视化
            if pending.get("result"):
                parts.append("**结果输出**\n" + str(pending["result"]))
            # 生成/保存的文件列表（保留结构，后续可扩展为图片/表格等可视化）
            output_files = pending.get("output_files") or []
            if output_files:
                lines = []
                for f in output_files:
                    path = f.get("path", f.get("name", ""))
                    name = f.get("name", path)
                    size_mb = f.get("size_mb")
                    size_bytes = f.get("size")
                    if size_mb is not None:
                        size_str = f" ({size_mb:.2f} MB)"
                    elif size_bytes:
                        size_str = f" ({size_bytes / (1024 * 1024):.2f} MB)"
                    else:
                        size_str = ""
                    lines.append(f"- `{name}` — {path}{size_str}")
                parts.append("**生成/保存的文件**\n" + "\n".join(lines))
            if pending.get("error"):
                parts.append("**错误**\n" + str(pending["error"]))
            # 失败时的执行日志（摘要）
            if not pending.get("success") and pending.get("output"):
                parts.append("**执行日志（摘要）**\n```\n" + str(pending["output"]).strip() + "\n```")
        if code and not any("代码" in p for p in parts):
            parts.append("**代码方案**\n```python\n" + code + "\n```")
        if not parts:
            parts.append(str(pending) if pending else "（无输出）")

    elif node_name == "tool_caller":
        pending = state.get("pending_contribution")
        report = state.get("final_report", "")
        if report:
            parts.append(report)
        if pending:
            parts.append(str(pending) if isinstance(pending, str) else str(pending))

    elif node_name == "critic":
        approved = state.get("is_approved", False)
        feedback = state.get("critique_feedback") or ""
        parts.append("**审核结果**: " + ("通过" if approved else "驳回"))
        if feedback:
            parts.append("\n**反馈**: " + feedback)

    elif node_name == "finalize":
        ans = state.get("final_answer", "")
        if ans:
            parts.append(ans)

    return "\n\n".join(parts).strip() or "（无文本输出）"


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="请输入您的任务描述，多智能体系统将进行规划、检索、代码生成/执行与审核，并在完成后展示最终答案。"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_query = (message.content or "").strip()
    if not user_query:
        await cl.Message(content="请输入有效任务描述。").send()
        return

    initial_state = build_initial_state(user_query)
    session_id = getattr(cl.context.session, "id", None) or "default"
    # 不使用 LangchainCallbackHandler，避免中间步骤以原始 JSON 展示；改为下面按节点解析后展示
    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }

    try:
        steps, final_state = await asyncio.to_thread(
            _run_graph_stream, initial_state, config_dict
        )
    except Exception as e:
        await cl.Message(content=f"执行出错: {e!s}").send()
        return

    # 按执行顺序展示每个 agent 的可读输出
    node_display_names = {
        "supervisor": "Supervisor（规划与调度）",
        "rag_researcher": "RAG Researcher（文献检索）",
        "code_dev": "Code Dev（代码生成/执行）",
        "tool_caller": "Tool Caller（数据分析）",
        "critic": "Critic（审核）",
        "finalize": "Finalize（汇总）",
    }
    for node_name, state in steps:
        display_name = node_display_names.get(node_name, node_name)
        text = _format_agent_output(node_name, state)
        # Code Dev 若有输出文件中的图片，附加到 Step 做可视化
        step_kwargs = {"name": display_name, "type": "agent"}
        if node_name == "code_dev":
            pending = state.get("pending_contribution") or {}
            if isinstance(pending, dict):
                image_els = _code_dev_image_elements(pending.get("output_files") or [])
                if image_els:
                    step_kwargs["elements"] = image_els
        async with cl.Step(**step_kwargs) as step:
            step.output = text

    if not final_state:
        await cl.Message(
            content="未得到最终状态（可能已达 recursion_limit）。请尝试简化任务或增大 recursion_limit。"
        ).send()
        return

    # 最终答案汇总（各 agent 输出已在上方按步骤展示）
    final_answer = final_state.get("final_answer", "")
    if final_answer:
        await cl.Message(content="**最终答案**\n\n" + final_answer).send()

    # 可选：将结果写入 results 目录便于复现与调试
    try:
        results_dir = _APP_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        plan = final_state.get("plan") or []
        plan_serializable = (
            [p.model_dump() if hasattr(p, "model_dump") else p for p in plan]
            if plan
            else []
        )
        out = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "final_answer": final_answer,
            "code_solution": final_state.get("code_solution", ""),
            "final_report": final_state.get("final_report", ""),
            "rag_context": final_state.get("rag_context", ""),
            "plan": plan_serializable,
        }
        out_path = results_dir / f"chainlit_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 忽略写入失败，不影响 UI
