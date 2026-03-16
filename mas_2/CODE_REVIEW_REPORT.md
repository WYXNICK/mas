# LangGraph 代码结构审查报告

## 一、重复代码问题

### 1.1 Critic Agent 中的步骤上下文构建逻辑重复

**位置**: `src/agents/critic/graph.py`

**问题**: 在 `check_umap_image`, `check_code`, `check_docs`, `check_db` 四个函数中，都有几乎完全相同的步骤上下文构建逻辑：

```python
# 构建步骤上下文信息
step_context_note = ""
if step_context:
    step_name = step_context.get("step_name", "")
    step_num = step_context.get("step_num", "")
    total_steps = step_context.get("total_steps", "")
    step_context_note = f"\n\n【重要上下文】这是多步骤任务中的一步：\n"
    if step_num and total_steps:
        step_context_note += f"- 当前步骤：步骤 {step_num}/{total_steps}\n"
    if step_name:
        step_context_note += f"- 步骤名称：{step_name}\n"
    step_context_note += "- 请只关注当前步骤的验收标准，不要要求完成整个任务的所有步骤。\n"
    step_context_note += "- 只要当前步骤的输出满足其验收标准，就应该通过审核。\n"

expected_output_note = ""
if expected_output:
    expected_output_note = f"\n\n【当前步骤的验收标准】\n{expected_output}\n请特别关注..."
```

**影响范围**: 
- `check_umap_image` (94-109行)
- `check_code` (144-159行)
- `check_docs` (195-210行)
- `check_db` (238-253行)

**建议**: 提取为公共函数 `_build_step_context_notes(step_context, expected_output, content_type="")`

---

### 1.2 从 State 提取步骤信息的逻辑重复

**位置**: 多个 agent 的 graph.py 文件

**问题**: 多个 agent 中都有类似的从 state 中提取 `current_step_input` 和 `current_step_expected_output` 的逻辑：

```python
current_step_input = state.get("current_step_input", "")
current_step_expected_output = state.get("current_step_expected_output", "")
```

**影响范围**:
- `code_dev/graph.py` (154-155行)
- `rag_researcher/graph.py` (88-89行)
- `tool_caller/nodes.py` (22-23行)

**建议**: 可以考虑在 `src/core/state.py` 中添加辅助函数，或者在每个 agent 的 state 类中添加属性方法

---

### 1.3 Main Graph 中的 Wrapper 函数模式重复

**位置**: `src/main.py`

**问题**: `wrap_rag_researcher`, `wrap_code_dev`, `wrap_tool_caller` 三个函数结构完全相同，只是 `last_worker` 的值不同：

```python
def wrap_rag_researcher(state: GlobalState) -> GlobalState:
    result = rag_agent_graph.invoke(state)
    return {**result, "last_worker": "rag_researcher"}

def wrap_code_dev(state: GlobalState) -> GlobalState:
    result = code_agent_graph.invoke(state)
    return {**result, "last_worker": "code_dev"}

def wrap_tool_caller(state: GlobalState) -> GlobalState:
    result = tool_caller_agent_graph.invoke(state)
    return {**result, "last_worker": "data_analyst"}
```

**建议**: 使用工厂函数或装饰器模式：
```python
def create_worker_wrapper(agent_graph, worker_name):
    def wrapper(state: GlobalState) -> GlobalState:
        result = agent_graph.invoke(state)
        return {**result, "last_worker": worker_name}
    return wrapper
```

---

## 二、结构性问题

### 2.1 Code Dev Agent 的 `execute_code` 函数过长

**位置**: `src/agents/code_dev/graph.py` (357-650行)

**问题**: `execute_code` 函数约 300 行，包含：
- 路径解析和验证逻辑（457-522行）
- Docker 执行逻辑（523-530行）
- 结果提取和错误处理（532-648行）

**建议**: 拆分为多个函数：
- `_determine_data_path(state)` - 确定数据路径
- `_execute_in_docker(code, requirements, data_path, output_dir)` - Docker 执行
- `_extract_execution_result(output_str)` - 提取执行结果
- `_handle_execution_error(output_str, result)` - 错误处理

---

### 2.2 Supervisor Agent 的 `_make_dynamic_decision` 函数较长

**位置**: `src/agents/supervisor/graph.py` (226-302行)

**问题**: 函数约 80 行，包含完整的 LLM 调用和错误处理逻辑

**建议**: 可以考虑将 prompt 构建逻辑提取为单独函数

---

### 2.3 Step Context 构建逻辑分散

**位置**: `src/agents/critic/graph.py`

**问题**: `review_contribution` 函数中构建 `step_context` 字典的逻辑（283-292行），与各个 check 函数中构建 `step_context_note` 的逻辑有重复

**建议**: 统一提取为公共函数，在 `review_contribution` 中构建 `step_context`，然后传递给各个 check 函数

---

## 三、代码组织问题

### 3.1 工具函数位置不明确

**位置**: `src/agents/critic/graph.py`

**问题**: `_normalize_base64_image` 函数（38-56行）是一个工具函数，但放在模块顶层，与其他审核函数混在一起

**建议**: 考虑创建 `src/agents/critic/utils.py` 或 `src/agents/critic/_utils/` 目录来存放工具函数

---

### 3.2 路径处理逻辑分散

**位置**: `src/agents/code_dev/graph.py`

**问题**: 路径处理逻辑分散在多个函数中：
- `parse_paths_from_query` (19-71行)
- `extract_paths_from_state` (74-119行)
- `execute_code` 中的路径确定逻辑 (457-522行)

**建议**: 考虑创建 `src/agents/code_dev/_utils/path_utils.py` 统一管理路径处理逻辑

---

## 四、潜在改进建议

### 4.1 类型提示一致性

**问题**: 部分函数缺少完整的类型提示，特别是返回类型

**建议**: 统一添加类型提示，提高代码可读性和 IDE 支持

---

### 4.2 错误处理统一性

**问题**: 不同 agent 中的错误处理方式不一致，有的使用 try-except，有的直接返回错误信息

**建议**: 考虑定义统一的错误处理策略或异常类

---

### 4.3 日志输出格式

**问题**: 不同 agent 的日志输出格式略有不同（有的用 `--- [Agent] ---`，有的用 `=== [Agent] ===`）

**建议**: 统一日志格式，或使用标准的 logging 模块

---

## 五、总结

### 优先级高的问题：
1. ✅ **Critic Agent 中的步骤上下文构建逻辑重复** - 影响 4 个函数，代码重复度高
2. ✅ **Main Graph 中的 Wrapper 函数模式重复** - 容易维护，影响 3 个函数

### 优先级中的问题：
3. ⚠️ **Code Dev Agent 的 execute_code 函数过长** - 影响可读性和维护性
4. ⚠️ **Step Context 构建逻辑分散** - 影响代码一致性

### 优先级低的问题：
5. 📝 **工具函数位置** - 代码组织优化
6. 📝 **类型提示和错误处理** - 代码质量提升

---

## 六、额外发现

### 6.1 工具函数组织良好

**位置**: `src/agents/code_dev/_utils/`

**发现**: Code Dev Agent 已经将工具函数组织在 `_utils` 目录下：
- `docker_path.py` - Docker 路径转换
- `base64_support.py` - Base64 图片处理

**建议**: 其他 agent 可以参考这种组织方式，将工具函数统一放在 `_utils` 目录下

---

### 6.2 Tool Caller Agent 的工具注册机制

**位置**: `src/agents/tool_caller/tools/__init__.py`

**发现**: 使用了动态工具注册机制，通过扫描模块自动注册工具，这是一个很好的设计模式

**建议**: 这种模式可以应用到其他需要插件化扩展的地方

---

## 七、建议的重构顺序

1. **第一步**: 提取 Critic Agent 中的公共函数 `_build_step_context_notes`
2. **第二步**: 重构 Main Graph 中的 wrapper 函数为工厂模式
3. **第三步**: 拆分 Code Dev Agent 的 `execute_code` 函数
4. **第四步**: 将 Critic Agent 的工具函数（如 `_normalize_base64_image`）移到 `_utils` 目录
5. **第五步**: 统一其他 agent 的工具函数组织方式（参考 Code Dev Agent 的模式）
