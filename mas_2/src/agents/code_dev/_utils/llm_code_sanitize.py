"""从 LLM 回复中得到的 Python 片段清理（不依赖 Docker）。"""
import re


def sanitize_llm_python_block(code: str) -> str:
    """去掉误插入的 Markdown 围栏行（如单独的 ```python），避免写入 code.py 后 SyntaxError。"""
    if not (code and code.strip()):
        return code
    out: list[str] = []
    for line in code.splitlines():
        if re.match(r"^```[\w.-]*\s*$", line.strip()):
            continue
        out.append(line)
    return "\n".join(out).strip("\n")
