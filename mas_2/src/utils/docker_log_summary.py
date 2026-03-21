"""
Docker / 容器 stdout 压缩：折叠 pip 噪声，保留 Traceback、Error 与 ===RESULT=== 相关段落。
"""
from __future__ import annotations

import re
from typing import List

# 典型 pip / setuptools 噪声行（整段折叠为一句摘要）
_PIP_NOISE_SUBSTRINGS = (
    "Collecting ",
    "Downloading ",
    "Installing ",
    "Requirement already satisfied",
    "Successfully installed",
    "WARNING: Running pip",
    "Looking in indexes",
    "Installing build dependencies",
    "Getting requirements to build",
    "Preparing metadata",
    "Building wheel for",
    "Created wheel for",
    "Stored in directory:",
    "DEPRECATION:",
    "Running setup.py",
    "Processing /tmp",
    "Downloading [",
    "  Downloading ",
    "  Installing ",
    "  Getting requirements",
    "  Preparing metadata",
    "  Building wheel",
    "  Created wheel",
    "pip install",
    "WARNING: You are using pip",
    "notice] A new release of pip",
    "Defaulting to user installation",
    "WARNING: The script",
    "Consider adding this directory to PATH",
    "not on PATH",
    "no-warn-script-location",
)

# pip/tqdm 进度条：含制表符进度或 x.x/x.x MB + MB/s
_RE_PIP_PROGRESS = re.compile(
    r"(?:━{4,}|╸{2,}|█{2,}).*(?:\d+/\d+\s*(?:MB|kB|KB|GB)|MB/s|it/s)",
    re.IGNORECASE,
)
_RE_MB_FRACTION = re.compile(r"\d+\.\d+/\d+\.\d+\s*(?:MB|kB|KB|GB)", re.IGNORECASE)
_RE_TQDM_BAR = re.compile(r"\d+%\|[^|]*\|")  # 100%|████| style

# 视为「错误/栈」保护区的行特征（整段保留）
_ERROR_LINE_HINTS = (
    "Traceback (most recent call last):",
    "Traceback (most recent call last)",
    "Error:",
    "Exception:",
    "===RESULT===",
)


def _is_pip_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return False
    if any(sub in s for sub in _PIP_NOISE_SUBSTRINGS):
        return True
    if s.startswith("WARNING:") and ("is installed in" in s or "not on PATH" in s):
        return True
    # tqdm / pip 默认进度条（Unicode 横条 + 大小 + 速度）
    if "━" in line and ("MB/s" in line or "kB/s" in line or _RE_MB_FRACTION.search(line)):
        return True
    if _RE_PIP_PROGRESS.search(line):
        return True
    if "MB/s" in line and _RE_MB_FRACTION.search(line):
        return True
    if _RE_TQDM_BAR.search(line):
        return True
    return False


def _is_traceback_start(line: str) -> bool:
    return "Traceback" in line and "most recent call" in line


def _find_traceback_end(lines: List[str], start: int) -> int:
    """从 start 起，返回 traceback 块最后一行下标（含）。"""
    n = len(lines)
    i = start
    last_meaningful = start
    in_trace = False
    while i < n:
        raw = lines[i]
        s = raw.rstrip()
        if _is_traceback_start(raw):
            in_trace = True
            last_meaningful = i
        elif in_trace:
            last_meaningful = i
            # 典型结尾：XxxError: message 或 Exception: ...
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*\s*:\s*", s):
                return i
            if s.startswith("Error:") or s.startswith("Exception:"):
                return i
            # 空行后若已出现过 File "，可能栈结束
            if not s and i > start + 2:
                # 再读一行判断是否错误行
                if i + 1 < n:
                    nxt = lines[i + 1].strip()
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*\s*:\s*", nxt):
                        return i + 1
                return last_meaningful
        i += 1
    return last_meaningful


def _find_result_block_end(lines: List[str], start: int) -> int:
    """包含 ===RESULT=== 的起始行，返回到第二个 === 闭合或行尾。"""
    n = len(lines)
    first = lines[start]
    if "===RESULT===" not in first:
        return start
    after = first.split("===RESULT===", 1)[-1]
    if "===" in after:
        return start
    # 多行情况：继续向下找 closing ===
    i = start + 1
    while i < n:
        if "===" in lines[i]:
            return i
        i += 1
    return n - 1


def _collapse_middle(lines: List[str], max_keep: int, head: int, tail: int) -> List[str]:
    """超过 max_keep 行时保留头尾（用于非保护段的二次压缩）。"""
    if len(lines) <= max_keep:
        return lines
    h = min(head, max_keep // 2)
    t = min(tail, max_keep - h)
    omitted = len(lines) - h - t
    if omitted <= 0:
        return lines
    mid_msg = f"... （已省略中间 {omitted} 行 stdout，完整日志见 output 字段或落盘文件）"
    return lines[:h] + [mid_msg] + lines[-t:]


def _merge_adjacent_pip_summaries(lines: List[str]) -> List[str]:
    """将连续的「【pip/依赖安装】已折叠 …」合并为一条，减少刷屏。"""
    if not lines:
        return lines
    out: List[str] = []
    i = 0
    n = len(lines)
    fold_re = re.compile(r"已折叠\s*(\d+)\s*行")
    succ_re = re.compile(r"Successfully installed 段\s*(\d+)\s*处")
    sat_re = re.compile(r"已满足依赖声明\s*(\d+)\s*条")

    while i < n:
        line = lines[i]
        if line.startswith("【pip/依赖安装】"):
            chunk = [line]
            j = i + 1
            while j < n and lines[j].startswith("【pip/依赖安装】"):
                chunk.append(lines[j])
                j += 1
            total_fold = 0
            total_succ = 0
            total_sat = 0
            for c in chunk:
                for m in fold_re.findall(c):
                    total_fold += int(m)
                for m in succ_re.findall(c):
                    total_succ += int(m)
                for m in sat_re.findall(c):
                    total_sat += int(m)
            parts = [f"【pip/依赖安装】共折叠 {total_fold} 行（含下载/进度条/pip 提示等）"]
            if total_succ:
                parts.append(f"Successfully installed 段 {total_succ} 处")
            if total_sat:
                parts.append(f"已满足依赖声明 {total_sat} 条")
            out.append("；".join(parts) + "。")
            i = j
            continue
        out.append(line)
        i += 1
    return out


def summarize_docker_stdout(
    raw: str,
    *,
    max_display_lines: int = 200,
    head_lines: int = 60,
    tail_lines: int = 60,
) -> str:
    """
    将原始容器输出压缩为适合 UI / Finalize 的短文本。

    - pip/依赖相关连续行 -> 单行中文摘要
    - Traceback 与典型 Error 结尾 -> 原样保留
    - 含 ===RESULT=== 的片段 -> 原样保留（与独立 result 字段并存，避免歧义可保留）
    - 其余过长时首尾保留 + 中间省略说明
    """
    if not raw or not raw.strip():
        return ""

    lines = raw.splitlines()
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if _is_traceback_start(line):
            end_tb = _find_traceback_end(lines, i)
            out.extend(lines[i : end_tb + 1])
            i = end_tb + 1
            continue

        if "===RESULT===" in line:
            end_r = _find_result_block_end(lines, i)
            out.extend(lines[i : end_r + 1])
            i = end_r + 1
            continue

        # 独立错误行（无 Traceback 前缀时仍保留上下文）
        stripped = line.strip()
        if stripped and any(h in stripped for h in _ERROR_LINE_HINTS if h != "===RESULT==="):
            if "Traceback" not in stripped:
                # 单行 Error / Exception
                if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*\s*:\s*", stripped) or stripped.startswith(
                    ("Error:", "Exception:")
                ):
                    out.append(line)
                    i += 1
                    continue

        if _is_pip_noise_line(line):
            j = i
            while j < n:
                if _is_pip_noise_line(lines[j]):
                    j += 1
                    continue
                if lines[j].strip() == "" and j + 1 < n and _is_pip_noise_line(lines[j + 1]):
                    j += 1
                    continue
                break
            count = j - i
            if count <= 0:
                out.append(line)
                i += 1
                continue
            # 统计 Successfully installed 包名数量（粗略）
            chunk = "\n".join(lines[i:j])
            n_success = len(re.findall(r"Successfully installed", chunk))
            n_satisfied = len(re.findall(r"Requirement already satisfied", chunk))
            msg_parts = [f"【pip/依赖安装】已折叠 {count} 行"]
            if n_success:
                msg_parts.append(f"Successfully installed 段 {n_success} 处")
            if n_satisfied:
                msg_parts.append(f"已满足依赖声明 {n_satisfied} 条")
            out.append("；".join(msg_parts) + "。")
            i = j
            continue

        out.append(line)
        i += 1

    text = "\n".join(out).strip()

    # 合并多条 pip 摘要为一行
    out_lines = text.splitlines()
    out_lines = _merge_adjacent_pip_summaries(out_lines)
    text = "\n".join(out_lines).strip()

    # 二次：整段仍过长则按行截断
    out_lines = text.splitlines()
    if len(out_lines) > max_display_lines:
        out_lines = _collapse_middle(out_lines, max_display_lines, head_lines, tail_lines)
        text = "\n".join(out_lines)

    # 字符级兜底（避免单行长 JSON 等）
    max_chars = 120_000
    if len(text) > max_chars:
        head_c = max_chars // 2
        tail_c = max_chars - head_c - 80
        text = (
            text[:head_c]
            + f"\n... （已省略 {len(text) - head_c - tail_c} 字符，完整内容见 output 字段）\n"
            + text[-tail_c:]
        )

    return text
