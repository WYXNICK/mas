"""
Chainlit 前端：MAS 多智能体系统测试与展示
运行: cd mas_2 && chainlit run app.py -w
"""
import asyncio
import hashlib
import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from datetime import datetime

# 可可视化为图片的扩展名（用于 Code Dev 输出文件）
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
_MAX_CODE_DEV_INLINE_IMAGES = 8
_THUMB_MAX_SIZE = (1024, 1024)

# 确保 mas_2 在路径中，以便 from src.main import graph 可用
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from langchain_core.messages import HumanMessage
import chainlit as cl

from src.utils.docker_log_summary import summarize_docker_stdout


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


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
        "current_step_skill_id": None,
        "next_worker": "rag_researcher",
        "last_worker": "",
        "final_report": "",
        "code_solution": "",
        "rag_context": "",
        "pending_contribution": None,
        "critique_feedback": None,
        "is_approved": False,
    }


# 主图顶层节点（与 src/main.py 中 add_node 名称一致；用于过滤 astream_events）
_MAIN_GRAPH_NODES = frozenset(
    {"supervisor", "rag_researcher", "code_dev", "tool_caller", "critic", "finalize"}
)

# Code Dev 子图内部节点（astream_events 上报时用于侧栏提示与子 Step）
_CODE_DEV_INNER_NODES = frozenset(
    {"generate_code", "self_reflection", "execute_code", "prepare_retry", "display_result"}
)
_CODE_DEV_INNER_PROGRESS = {
    "generate_code": "Code Dev · 正在生成代码…",
    "self_reflection": "Code Dev · 自我检查…",
    "execute_code": "Code Dev · 正在 Docker 中执行…",
    "prepare_retry": "Code Dev · 准备重试…",
    "display_result": "Code Dev · 正在整理展示…",
}
_CODE_DEV_INNER_DONE = {
    "generate_code": "Code Dev · 代码已生成",
    "self_reflection": "Code Dev · 自检完成",
    "execute_code": "Code Dev · 执行阶段结束",
    "prepare_retry": "Code Dev · 重试准备完成",
    "display_result": "Code Dev · 展示完成",
}


def _code_dev_inner_from_event(event: dict) -> str | None:
    name = event.get("name")
    if isinstance(name, str) and name in _CODE_DEV_INNER_NODES:
        return name
    meta = event.get("metadata") or {}
    lg_node = meta.get("langgraph_node")
    if isinstance(lg_node, str) and lg_node in _CODE_DEV_INNER_NODES:
        return lg_node
    ns = str(meta.get("langgraph_checkpoint_ns") or "")
    for n in sorted(_CODE_DEV_INNER_NODES, key=len, reverse=True):
        if n in ns:
            return n
    return None


def _run_graph_stream(initial_state: dict, config: dict) -> tuple[list[tuple[str, dict]], dict | None]:
    """执行 graph.stream，返回 (每步 (node_name, state) 列表, 最终 state)。供测试或回退使用。"""
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


def _chunk_to_text(chunk) -> str:
    """从 LangChain/LangGraph 流式 chunk 中提取可展示文本。"""
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else str(chunk)


def _resolve_image_path_for_display(src: Path) -> Path:
    """大图时可选生成 JPEG 缩略图（需 Pillow + MAS_CHAINLIT_IMAGE_THUMBS=1），默认直接用原图以免变慢。"""
    if not _env_flag("MAS_CHAINLIT_IMAGE_THUMBS"):
        return src
    try:
        from PIL import Image
    except ImportError:
        return src
    try:
        if src.stat().st_size < 350_000:
            return src
    except OSError:
        return src
    cache_dir = _APP_DIR / ".chainlit_thumb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(src.resolve()).encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{key}.jpg"
    try:
        if cache_file.exists() and cache_file.stat().st_mtime >= src.stat().st_mtime:
            return cache_file
        im = Image.open(src)
        im = im.convert("RGB")
        im.thumbnail(_THUMB_MAX_SIZE)
        im.save(cache_file, "JPEG", quality=85, optimize=True)
        return cache_file
    except Exception:
        return src


def _code_dev_image_elements(output_files: list) -> list:
    """从 output_files 中筛选图片，最多内联 _MAX_CODE_DEV_INLINE_IMAGES 张；大图用缩略图路径。"""
    elements = []
    for f in output_files or []:
        if len(elements) >= _MAX_CODE_DEV_INLINE_IMAGES:
            break
        path = f.get("path") or f.get("name")
        name = f.get("name") or (Path(path).name if path else "image")
        if not path or not isinstance(path, str):
            continue
        path = Path(path)
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        display_path = _resolve_image_path_for_display(path)
        try:
            elements.append(
                cl.Image(
                    path=str(display_path.resolve()),
                    name=name,
                    display="inline",
                    size="medium",
                )
            )
        except Exception:
            continue
    return elements


def _count_output_images(output_files: list) -> tuple[list[Path], int]:
    """返回 (存在的图片路径列表, 总数)。"""
    found: list[Path] = []
    for f in output_files or []:
        path = f.get("path") or f.get("name")
        if not path or not isinstance(path, str):
            continue
        p = Path(path)
        if p.exists() and p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
            found.append(p)
    return found, len(found)


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
                parts.append("**结果输出**（===RESULT=== 提取）\n" + str(pending["result"]))
            if pending.get("error"):
                parts.append("**错误**\n" + str(pending["error"]))

            raw_out = (pending.get("output") or "").strip()
            tail_out = (pending.get("output_tail") or "").strip()
            disp = (pending.get("output_display") or "").strip()
            if not disp and raw_out:
                disp = summarize_docker_stdout(str(raw_out))
            if not disp and tail_out:
                disp = summarize_docker_stdout(str(tail_out))
            if disp:
                parts.append("**压缩执行日志**\n```\n" + disp + "\n```")
            log_disk = (pending.get("output_log_path") or "").strip()
            if log_disk:
                parts.append(f"**完整执行日志文件**（服务端）：`{log_disk}`")

            # 子图 execute 节点已落盘时带 output_log_path；此处仅作兼容旧 state（仍含 output）的补写
            if (
                _env_flag("MAS_SAVE_FULL_EXEC_LOG")
                and raw_out
                and not (pending.get("output_log_path") or "").strip()
            ):
                results_dir = _APP_DIR / "results"
                try:
                    results_dir.mkdir(parents=True, exist_ok=True)
                    log_name = f"exec_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.log"
                    log_path = results_dir / log_name
                    log_path.write_text(str(raw_out), encoding="utf-8")
                    parts.append(f"**完整执行日志** 已落盘：`{log_path}`")
                except OSError:
                    pass

            if _env_flag("MAS_CHAINLIT_SHOW_FULL_EXEC_LOG"):
                if raw_out:
                    parts.append("**完整执行日志**\n```\n" + str(raw_out).strip() + "\n```")
                elif tail_out:
                    parts.append(
                        "**完整执行日志**（默认未保留整段 stdout；以下为尾部）\n```\n"
                        + str(tail_out).strip()
                        + "\n```\n"
                        "可设置 `MAS_KEEP_FULL_EXEC_OUTPUT_IN_STATE=1` 或在 state 中查看 `output_log_path`。"
                    )

            # 生成/保存的文件列表（保留结构，后续可扩展为图片/表格等可视化）
            output_files = pending.get("output_files") or []
            _img_paths, n_img = _count_output_images(output_files)
            if n_img > _MAX_CODE_DEV_INLINE_IMAGES:
                rest = _img_paths[_MAX_CODE_DEV_INLINE_IMAGES:]
                parts.append(
                    f"**图片说明**：界面内仅内联前 {_MAX_CODE_DEV_INLINE_IMAGES} 张，另有 {len(rest)} 张未内联：\n"
                    + "\n".join(f"- `{p}`" for p in rest)
                )
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
        if code and not any("代码" in p for p in parts):
            code_txt = code.get("code", "") if isinstance(code, dict) else str(code)
            if code_txt.strip():
                parts.append("**代码方案**\n```python\n" + code_txt + "\n```")
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


NODE_DISPLAY_NAMES = {
    "supervisor": "Supervisor（规划与调度）",
    "rag_researcher": "RAG Researcher（文献检索）",
    "code_dev": "Code Dev（代码生成/执行）",
    "tool_caller": "Tool Caller（数据分析）",
    "critic": "Critic（审核）",
    "finalize": "Finalize（汇总）",
}

# 侧栏固定拓扑顺序（与 UI 布局一致）
_MAIN_GRAPH_AGENT_ORDER: tuple[str, ...] = (
    "supervisor",
    "rag_researcher",
    "code_dev",
    "tool_caller",
    "critic",
    "finalize",
)

_SNAPSHOT_TEXT_MAX = 800
_SNAPSHOT_CODE_PREVIEW_MAX = 480


def _truncate_text(s: str, max_len: int = _SNAPSHOT_TEXT_MAX) -> str:
    t = (s or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _snapshot_for_agent(node_id: str, state: dict) -> dict:
    """从 GlobalState 抽取单节点可读快照（可 JSON 序列化），供侧栏 CustomElement 展示。"""
    title = NODE_DISPLAY_NAMES.get(node_id, node_id)
    lines: list[str] = []

    if node_id == "supervisor":
        plan = state.get("plan") or []
        lines.append(f"计划步骤数：{len(plan)}")
        lines.append(f"current_step_index：{state.get('current_step_index', 0)}")
        nw = state.get("next_worker") or ""
        if nw:
            lines.append(f"next_worker：{nw}")
        if plan:
            p0 = plan[0]
            if isinstance(p0, dict):
                nm, sk = p0.get("name", ""), p0.get("skill_id") or "—"
            else:
                nm, sk = getattr(p0, "name", ""), getattr(p0, "skill_id", None) or "—"
            lines.append(f"首步：{nm}（skill：{sk}）")

    elif node_id == "rag_researcher":
        rag = state.get("rag_context", "")
        if rag:
            lines.append(f"rag_context（摘要）：{_truncate_text(str(rag), 600)}")
        else:
            lines.append("尚无 rag_context")

    elif node_id == "code_dev":
        pending = state.get("pending_contribution")
        if isinstance(pending, dict):
            code = pending.get("code")
            if code:
                lines.append(
                    f"代码预览：{_truncate_text(str(code), _SNAPSHOT_CODE_PREVIEW_MAX)}"
                )
            err = pending.get("error")
            if err:
                lines.append(f"错误：{_truncate_text(str(err), 400)}")
            res = pending.get("result")
            if res:
                lines.append(f"结果摘要：{_truncate_text(str(res), 500)}")
            if pending.get("output_files"):
                lines.append(f"输出文件数：{len(pending['output_files'])}")
        elif pending is not None:
            lines.append(_truncate_text(str(pending), 500))
        cs = state.get("code_solution")
        if isinstance(cs, dict) and cs.get("code") and not any("代码预览" in x for x in lines):
            lines.append(
                f"code_solution 预览：{_truncate_text(str(cs.get('code')), _SNAPSHOT_CODE_PREVIEW_MAX)}"
            )

    elif node_id == "tool_caller":
        report = state.get("final_report", "")
        if report:
            lines.append(_truncate_text(str(report), 650))
        pending = state.get("pending_contribution")
        if pending and isinstance(pending, str):
            lines.append(_truncate_text(pending, 400))

    elif node_id == "critic":
        lines.append("审核结果：" + ("通过" if state.get("is_approved") else "驳回"))
        fb = state.get("critique_feedback") or ""
        if fb:
            lines.append(f"反馈：{_truncate_text(str(fb), 650)}")

    elif node_id == "finalize":
        ans = state.get("final_answer", "")
        if ans:
            lines.append(f"final_answer：{_truncate_text(str(ans), 720)}")
        else:
            lines.append("尚无 final_answer")

    if not lines:
        lines.append("（本轮无摘要产出）")

    return {"title": title, "lines": lines}


def _format_visited_path(visited: list[str]) -> str:
    if not visited:
        return "（尚无）"
    return " → ".join(NODE_DISPLAY_NAMES.get(n, n) for n in visited)


def _default_main_graph_sidebar_props() -> dict:
    """会话初始 / 无快照时「打开主图」按钮使用的占位内容。"""
    return {
        "agentOrder": list(_MAIN_GRAPH_AGENT_ORDER),
        "activeId": None,
        "visitedSequence": [],
        "snapshots": {},
        "statusLine": "尚未开始任务。发送一条消息后，此处会显示当前执行节点。",
        "pathLine": f"已完成路径：{_format_visited_path([])}",
        "note": "",
    }


def build_graph_sidebar_props(
    *,
    current_node: str | None,
    visited: list[str],
    astream_mode: bool,
    snapshots: dict[str, dict],
    note: str | None = None,
) -> dict:
    """侧栏 CustomElement MainGraphFlow：固定拓扑 + 高亮 + 各 Agent 快照。"""
    if astream_mode:
        if current_node:
            cur = NODE_DISPLAY_NAMES.get(current_node, current_node)
            status = f"当前执行：{cur}"
        else:
            status = "当前执行：（节点间隙 / 等待下一节点）"
    else:
        last = visited[-1] if visited else None
        last_d = NODE_DISPLAY_NAMES.get(last, last) if last else "—"
        status = (
            "说明：astream_events 不可用，仅在节点完成后更新。\n"
            f"最近完成：{last_d}"
        )

    return {
        "agentOrder": list(_MAIN_GRAPH_AGENT_ORDER),
        "activeId": current_node,
        "visitedSequence": list(visited),
        "snapshots": snapshots,
        "statusLine": status,
        "pathLine": f"已完成路径：{_format_visited_path(visited)}",
        "note": (note or "").strip(),
    }


class GraphSidebarController:
    """Chainlit ElementSidebar：固定主图卡片 + 当前 Agent + 各节点状态快照。"""

    __slots__ = (
        "astream_mode",
        "current_node",
        "visited",
        "snapshots",
        "code_dev_inner_line",
        "_push_seq",
    )

    def __init__(self, *, astream_mode: bool) -> None:
        self.astream_mode = astream_mode
        self.current_node: str | None = None
        self.visited: list[str] = []
        self.snapshots: dict[str, dict] = {}
        self.code_dev_inner_line: str | None = None
        # Chainlit：set_elements 在「与上次相同的 key」下不会替换内容，导致主图不随执行更新
        self._push_seq = 0

    def mark_start(self, name: str) -> None:
        if name in _MAIN_GRAPH_NODES:
            self.current_node = name

    def mark_end(self, name: str) -> None:
        if name in _MAIN_GRAPH_NODES:
            self.visited.append(name)
            self.current_node = None

    def mark_completed_only(self, name: str) -> None:
        if name in _MAIN_GRAPH_NODES:
            self.visited.append(name)

    def merge_snapshot(self, node_id: str, state: dict) -> None:
        if node_id not in _MAIN_GRAPH_NODES:
            return
        self.snapshots[node_id] = _snapshot_for_agent(node_id, state)
        if node_id == "code_dev":
            self.code_dev_inner_line = None

    def set_code_dev_inner_line(self, line: str | None) -> None:
        self.code_dev_inner_line = (line or "").strip() or None

    def _snapshots_for_props(self) -> dict[str, dict]:
        out: dict[str, dict] = {k: dict(v) for k, v in self.snapshots.items()}
        if self.code_dev_inner_line:
            base = out.get("code_dev") or {
                "title": NODE_DISPLAY_NAMES["code_dev"],
                "lines": ["Code Dev 主节点运行中…"],
            }
            merged = dict(base)
            merged["innerLine"] = self.code_dev_inner_line
            out["code_dev"] = merged
        return out

    async def push(self, *, note: str | None = None) -> None:
        props = build_graph_sidebar_props(
            current_node=self.current_node,
            visited=list(self.visited),
            astream_mode=self.astream_mode,
            snapshots=self._snapshots_for_props(),
            note=note,
        )
        cl.user_session.set("main_graph_sidebar_props", props)
        self._push_seq += 1
        await cl.ElementSidebar.set_title("主图执行状态")
        await cl.ElementSidebar.set_elements(
            [cl.CustomElement(name="MainGraphFlow", props=props)],
            key=f"main_graph_exec_{self._push_seq}",
        )

    async def show_placeholder(self) -> None:
        await self.push(note="状态：工作流已启动，等待图事件…")

    async def set_idle_title(self) -> None:
        """任务结束后更新标题；保留最后一次 push 的图与路径，便于对照聊天 Steps。"""
        await cl.ElementSidebar.set_title("主图 — 就绪")


@cl.action_callback("show_main_graph_sidebar")
async def on_show_main_graph_sidebar(action: cl.Action) -> None:
    """聊天消息中的「打开主图」按钮：用最近一次侧栏快照重新打开 ElementSidebar。"""
    props = cl.user_session.get("main_graph_sidebar_props") or _default_main_graph_sidebar_props()
    await cl.ElementSidebar.set_title("主图执行状态")
    await cl.ElementSidebar.set_elements(
        [cl.CustomElement(name="MainGraphFlow", props=props)],
        key=f"main_graph_reopen_{uuid.uuid4().hex}",
    )


async def _emit_agent_step(node_name: str, state: dict) -> None:
    """将单个 agent 节点的可读输出发到 Chainlit（一个 Step）。"""
    display_name = NODE_DISPLAY_NAMES.get(node_name, node_name)
    text = _format_agent_output(node_name, state)
    step_kwargs: dict = {"name": display_name, "type": "agent"}
    if node_name == "code_dev":
        pending = state.get("pending_contribution") or {}
        if isinstance(pending, dict):
            image_els = _code_dev_image_elements(pending.get("output_files") or [])
            if image_els:
                step_kwargs["elements"] = image_els
    async with cl.Step(**step_kwargs) as step:
        step.output = text


async def _emit_code_dev_inner_step(inner_name: str, state: dict | None) -> None:
    """Code Dev 子图单节点完成时发一条轻量子 Step（不替代顶层 Code Dev Step）。"""
    title = _CODE_DEV_INNER_DONE.get(inner_name, inner_name)
    body = f"**{title}**"
    if inner_name == "execute_code" and isinstance(state, dict) and "success" in state:
        body += f"\n本轮 success 字段：`{state.get('success')}`"
    async with cl.Step(name=f"Code Dev / {inner_name}", type="tool") as step:
        step.output = body


async def _drain_code_dev_log_queue(log_q: queue.Queue, graph_done: asyncio.Event) -> None:
    """从 executor 线程 put 的队列读出 Docker 日志块，流式发到 Chainlit。"""
    msg = cl.Message(content="**Docker 执行日志**（实时）\n\n")
    await msg.send()
    idle_after_done = 0
    while True:
        try:
            chunk = log_q.get_nowait()
            idle_after_done = 0
            await msg.stream_token(chunk)
            continue
        except queue.Empty:
            pass
        if graph_done.is_set():
            idle_after_done += 1
            if idle_after_done > 20:
                break
        await asyncio.sleep(0.04)


async def _run_graph_incremental_queue(
    graph,
    initial_state: dict,
    config_dict: dict,
    sidebar: GraphSidebarController | None = None,
) -> tuple[dict | None, bool]:
    """
    同步 graph.stream 放在后台线程，每完成一个顶层节点即通过队列交给主协程发 Step。
    不流式输出 LLM token，但不必等整条图跑完才看到各节点结果。
    返回 (final_state, streamed_finalize_tokens)。
    """
    q: queue.Queue = queue.Queue()
    err_holder: list[BaseException] = []

    def worker() -> None:
        try:
            for step in graph.stream(initial_state, config=config_dict):
                for node_name, state in step.items():
                    q.put((node_name, state))
            q.put(None)
        except BaseException as e:
            err_holder.append(e)
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    final_state: dict | None = None
    while True:
        item = await asyncio.to_thread(q.get)
        if item is None:
            break
        node_name, state = item
        final_state = state
        if sidebar is not None:
            sidebar.mark_completed_only(node_name)
            sidebar.merge_snapshot(node_name, state)
            await sidebar.push()
        await _emit_agent_step(node_name, state)
    if err_holder:
        raise err_holder[0]
    return final_state, False


async def _run_graph_astream_events(
    graph,
    initial_state: dict,
    config_dict: dict,
    sidebar: GraphSidebarController | None = None,
) -> tuple[dict | None, bool]:
    """
    使用 LangGraph astream_events：顶层节点完成即展示 Step；finalize 内 LLM 的 token 流式到一条 Message。
    若当前环境事件结构不兼容，请回退到 _run_graph_incremental_queue。
    """
    final_state: dict | None = None
    streamed_finalize_tokens = False
    streaming_msg = None

    async for event in graph.astream_events(
        initial_state,
        config_dict,
        version="v2",
    ):
        ev = event.get("event")
        if ev == "on_chain_start":
            inner = _code_dev_inner_from_event(event)
            if inner:
                if sidebar is not None:
                    sidebar.set_code_dev_inner_line(
                        _CODE_DEV_INNER_PROGRESS.get(inner, inner)
                    )
                    await sidebar.push()
                continue
            name = event.get("name")
            if name not in _MAIN_GRAPH_NODES:
                continue
            if sidebar is not None:
                sidebar.mark_start(name)
                await sidebar.push()
        elif ev == "on_chain_end":
            inner = _code_dev_inner_from_event(event)
            if inner:
                data = event.get("data")
                out = data.get("output") if isinstance(data, dict) else None
                await _emit_code_dev_inner_step(
                    inner, out if isinstance(out, dict) else None
                )
                if sidebar is not None:
                    sidebar.set_code_dev_inner_line(
                        _CODE_DEV_INNER_DONE.get(inner, inner)
                    )
                    await sidebar.push()
                continue
            name = event.get("name")
            if name not in _MAIN_GRAPH_NODES:
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            out = data.get("output")
            if not isinstance(out, dict):
                continue
            final_state = out
            if sidebar is not None:
                sidebar.merge_snapshot(name, out)
                sidebar.mark_end(name)
                await sidebar.push()
            # finalize 若已做 token 流式，避免 Step 再贴一整段重复的最终答案
            if name == "finalize" and streamed_finalize_tokens:
                continue
            await _emit_agent_step(name, out)
        elif ev == "on_chat_model_stream":
            meta = event.get("metadata") or {}
            if meta.get("langgraph_node") != "finalize":
                continue
            data = event.get("data") or {}
            chunk = data.get("chunk")
            token = _chunk_to_text(chunk)
            if not token:
                continue
            streamed_finalize_tokens = True
            if streaming_msg is None:
                streaming_msg = cl.Message(content="**最终答案**（生成中…）\n\n")
                await streaming_msg.send()
            await streaming_msg.stream_token(token)

    return final_state, streamed_finalize_tokens


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("main_graph_sidebar_props", _default_main_graph_sidebar_props())
    await cl.Message(
        content=(
            "请输入您的任务描述，多智能体系统将进行规划、检索、代码生成/执行与审核，并在完成后展示最终答案。\n\n"
            "点击下方 **打开主图** 可在侧栏查看固定主图、运行高亮与各 Agent 状态摘要（执行过程中会自动刷新）。"
        ),
        actions=[
            cl.Action(name="show_main_graph_sidebar", label="打开主图", payload={}),
        ],
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
    configurable: dict = {"thread_id": session_id}
    graph_done = asyncio.Event()
    log_q: queue.Queue | None = None
    log_pump_task: asyncio.Task | None = None
    if _env_flag("MAS_CHAINLIT_STREAM_DOCKER_LOG"):
        log_q = queue.Queue()
        configurable["code_dev_log_queue"] = log_q

        async def _log_pump_runner() -> None:
            try:
                if log_q is not None:
                    await _drain_code_dev_log_queue(log_q, graph_done)
            except Exception as ex:
                print(f"[chainlit] Docker 日志流: {ex!s}")

        log_pump_task = asyncio.create_task(_log_pump_runner())

    config_dict = {
        "configurable": configurable,
        "recursion_limit": 20,
    }

    for candidate in (_APP_DIR, Path.cwd()):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from src.main import graph

    final_state = None
    streamed_finalize = False
    sidebar: GraphSidebarController | None = None
    try:
        if hasattr(graph, "astream_events"):
            sidebar = GraphSidebarController(astream_mode=True)
            try:
                await sidebar.show_placeholder()
                final_state, streamed_finalize = await _run_graph_astream_events(
                    graph, initial_state, config_dict, sidebar=sidebar
                )
            except Exception as e:
                print(f"[chainlit] astream_events 不可用或失败，回退为按节点增量展示: {e!s}")
                sidebar = GraphSidebarController(astream_mode=False)
                await sidebar.show_placeholder()
                final_state, streamed_finalize = await _run_graph_incremental_queue(
                    graph, initial_state, config_dict, sidebar=sidebar
                )
        else:
            sidebar = GraphSidebarController(astream_mode=False)
            await sidebar.show_placeholder()
            final_state, streamed_finalize = await _run_graph_incremental_queue(
                graph, initial_state, config_dict, sidebar=sidebar
            )
    except Exception as e:
        if sidebar is not None:
            sidebar.current_node = None
            try:
                await sidebar.push(note=f"**执行出错**：`{e!s}`")
            except Exception:
                pass
        await cl.Message(content=f"执行出错: {e!s}").send()
        return
    finally:
        graph_done.set()
        if log_pump_task is not None:
            try:
                await asyncio.sleep(0.3)
                log_pump_task.cancel()
                await log_pump_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if sidebar is not None:
            try:
                await sidebar.set_idle_title()
            except Exception:
                pass

    if not final_state:
        await cl.Message(
            content="未得到最终状态（可能已达 recursion_limit）。请尝试简化任务或增大 recursion_limit。"
        ).send()
        return

    # 最终答案：若已在 astream_events 中对 finalize 做过 token 流式，则不再整段粘贴
    final_answer = final_state.get("final_answer", "")
    if final_answer and not streamed_finalize:
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
        cs = final_state.get("code_solution", "")
        if isinstance(cs, dict) and not _env_flag("MAS_CHAINLIT_RESULT_JSON_FULL_OUTPUT"):
            cs = {k: v for k, v in cs.items() if k != "output"}
            cs["_note"] = "完整 output 已从 JSON 省略；设 MAS_CHAINLIT_RESULT_JSON_FULL_OUTPUT=1 可写入"

        out = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "final_answer": final_answer,
            "code_solution": cs,
            "final_report": final_state.get("final_report", ""),
            "rag_context": final_state.get("rag_context", ""),
            "plan": plan_serializable,
        }
        out_path = results_dir / f"chainlit_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 忽略写入失败，不影响 UI
