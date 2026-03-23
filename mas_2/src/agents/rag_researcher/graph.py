"""
RAG Researcher Agent 子图
负责向量检索和文档查询
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Optional
from langgraph.graph import StateGraph, START, END
from .state import RAGAgentState
from src.utils.project_paths import resolve_chroma_persist_path

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("警告: chromadb 或 sentence-transformers 未安装，RAG 功能将受限")

# ---------------------------
# 配置
# ---------------------------

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "default_collection")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
# 为 1 时：若当前步骤有 skill_id，则优先按 metadata skill_id 过滤；无结果时回退为全库检索
RAG_SKILL_FILTER = os.getenv("RAG_SKILL_FILTER", "1").strip().lower() not in ("0", "false", "no", "off")


def _resolve_chroma_path(base_path: str, collection_name: str) -> str:
    """兼容两种布局:
    1) 旧布局: chroma_db/ + collection
    2) 新布局: chroma_db/<collection_name>/ + collection
    """
    base = Path(base_path).resolve()
    if base.name == collection_name and (base / "chroma.sqlite3").exists():
        return str(base)

    candidate = base / collection_name
    if candidate.exists() and (candidate / "chroma.sqlite3").exists():
        return str(candidate)

    return str(base)


# ---------------------------
# 嵌入与向量库
# ---------------------------

@lru_cache(maxsize=1)
def _get_embedder():
    if not CHROMA_AVAILABLE:
        return None
    return SentenceTransformer(EMBED_MODEL_NAME)


class _STEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input: List[str]):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input or [])
        if not texts:
            return []
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_documents(self, input: List[str]):
        return self.__call__(input)

    def embed_query(self, input: str):
        if isinstance(input, list):
            text = input[0] if input else ""
        else:
            text = input or ""
        if not text:
            return []
        return self.model.encode([text], normalize_embeddings=True).tolist()

    def name(self):
        return "sentence-transformer"


@lru_cache(maxsize=1)
def _get_collection():
    if not CHROMA_AVAILABLE:
        return None
    
    embedder = _get_embedder()
    if embedder is None:
        return None
    
    resolved_path = _resolve_chroma_path(
        str(resolve_chroma_persist_path()), CHROMA_COLLECTION
    )
    client = chromadb.PersistentClient(path=resolved_path)
    embedding_fn = _STEmbeddingFunction(embedder)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
    )


def _vector_search(
    query: str,
    top_k: int = TOP_K,
    skill_id: Optional[str] = None,
) -> List[str]:
    """向量检索；可选按 chunk metadata skill_id 过滤（与 parse_docs --skill-id 入库一致）。"""
    collection = _get_collection()
    if collection is None:
        return ["向量库未配置，请检查 chromadb 和 sentence-transformers 是否已安装"]

    sid = (skill_id or "").strip() or None

    def _run(where: Optional[dict]) -> List[str]:
        kwargs = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = collection.query(**kwargs)
        docs = result.get("documents") or []
        flat = docs[0] if docs else []
        distances = (result.get("distances") or [[]])[0]
        if flat and distances and len(flat) == len(distances):
            ranked = sorted(zip(flat, distances), key=lambda x: x[1])
            flat = [doc for doc, _ in ranked]
        return flat

    try:
        flat: List[str] = []
        if RAG_SKILL_FILTER and sid:
            flat = _run({"skill_id": sid})
        if not flat:
            flat = _run(None)
        if not flat:
            return ["未找到相关文献片段"]
        return flat
    except Exception as e:
        return [f"检索出错: {str(e)}"]


def search_documents(state: RAGAgentState) -> RAGAgentState:
    """
    检索文档节点
    从向量库中检索相关文档
    """
    # 优先使用当前步骤的输入作为查询内容
    current_step_input = state.get("current_step_input", "")
    current_step_expected_output = state.get("current_step_expected_output", "")
    
    # 构建查询：如果有当前步骤输入，使用它；否则使用search_query或user_query
    if current_step_input:
        query = current_step_input
        print(f"--- [RAG Researcher] 正在查询向量库（根据计划步骤）: {query[:100]}... ---")
    else:
        query = state.get("search_query") or state.get("user_query", "")
        print(f"--- [RAG Researcher] 正在查询向量库: {query} ---")
    
    # 如果有预期输出，在日志中显示
    if current_step_expected_output:
        print(f"  --> 预期输出要求: {current_step_expected_output[:100]}...")
    
    runtime_top_k = state.get("rag_top_k", TOP_K)
    try:
        runtime_top_k = int(runtime_top_k)
    except Exception:
        runtime_top_k = TOP_K
    runtime_top_k = max(1, runtime_top_k)

    docs = _vector_search(
        query,
        top_k=runtime_top_k,
        skill_id=state.get("current_step_skill_id"),
    )
    
    # 更新状态
    state["retrieved_docs"] = docs
    state["doc_count"] = len(docs)
    
    # 将结果写入 pending_contribution
    state["pending_contribution"] = docs
    
    # 同时更新 rag_context
    state["rag_context"] = "\n\n".join(docs)
    
    print(f"  --> 检索到 {len(docs)} 条文档")

    print(f"  --> 文档内容（前200字符）:\n{state['rag_context'][:200]}...")
    
    return state


# 构建子图
workflow = StateGraph(RAGAgentState)

# 添加节点
workflow.add_node("search_documents", search_documents)

# 定义边
workflow.add_edge(START, "search_documents")
workflow.add_edge("search_documents", END)

# 编译子图
rag_agent_graph = workflow.compile()

