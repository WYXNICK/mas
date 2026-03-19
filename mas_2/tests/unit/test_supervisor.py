"""
Supervisor Agent 单元测试
使用 Mock 模拟 LLM 调用
"""
import pytest
from unittest.mock import patch, MagicMock
from src.agents.supervisor.graph import supervisor_agent_graph
from src.agents.supervisor.graph import RouteDecision
from src.core.state import PlanStep


def _build_single_step_plan() -> list[PlanStep]:
    """构造最小可执行计划，避免单测触发真实计划生成。"""
    return [
        PlanStep(
            step_id=1,
            name="生成代码",
            description="生成并执行一段测试代码",
            input_files=[],
            output_files=["./result/step_1_test.txt"],
            acceptance_criteria="输出测试结果",
        )
    ]


def test_supervisor_decision_with_mock(supervisor_state):
    """
    测试 Supervisor Agent 的决策逻辑
    
    使用 @patch 模拟 LLM 的返回，验证路由决策是否正确
    """
    # 准备测试状态
    state = supervisor_state.copy()
    state["user_query"] = "需要生成代码"
    state["plan"] = _build_single_step_plan()
    state["current_step_index"] = 0
    
    # 创建模拟的 RouteDecision 对象
    mock_decision = RouteDecision(
        next_worker="code_dev",
        reasoning="用户需要生成代码，应该调用 code_dev worker"
    )
    
    # Mock LLM 的 with_structured_output 方法
    with patch('src.agents.supervisor.graph.llm') as mock_llm:
        # 创建模拟的 chain 对象
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_decision
        
        # 设置 with_structured_output 返回 mock_chain
        mock_llm.with_structured_output.return_value = mock_chain
        
        # 执行 Supervisor Agent 子图
        result = supervisor_agent_graph.invoke(state)
        
        # 断言：next_worker 字段确实变成了 "code_dev"
        assert result.get("next_worker") == "code_dev", \
            f"期望 next_worker='code_dev'，实际为 '{result.get('next_worker')}'"

        # 断言：当前步骤上下文已写入状态
        assert result.get("current_step_input") == "生成并执行一段测试代码"
        assert result.get("current_step_expected_output") == "输出测试结果"
        assert result.get("current_step_file_paths") == {
            "input_files": [],
            "output_files": ["./result/step_1_test.txt"],
        }
        
        # 验证 LLM 被调用
        mock_llm.with_structured_output.assert_called_once()
        mock_chain.invoke.assert_called_once()


def test_supervisor_decision_finish(supervisor_state):
    """
    测试 Supervisor 决定结束任务的情况
    """
    state = supervisor_state.copy()
    state["user_query"] = "任务已完成"
    state["rag_context"] = "已有上下文"
    state["code_solution"] = "已有代码"
    state["plan"] = _build_single_step_plan()
    state["current_step_index"] = 0
    
    # 创建模拟的 RouteDecision 对象（决定结束）
    mock_decision = RouteDecision(
        next_worker="FINISH",
        reasoning="所有任务已完成，可以结束"
    )
    
    with patch('src.agents.supervisor.graph.llm') as mock_llm:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_decision
        mock_llm.with_structured_output.return_value = mock_chain
        
        result = supervisor_agent_graph.invoke(state)
        
        # 断言：next_worker 应该是 "FINISH"
        assert result.get("next_worker") == "FINISH", \
            f"期望 next_worker='FINISH'，实际为 '{result.get('next_worker')}'"


def test_supervisor_decision_rag_researcher(supervisor_state):
    """
    测试 Supervisor 决定调用 RAG Researcher 的情况
    """
    state = supervisor_state.copy()
    state["user_query"] = "查询相关文献"
    state["rag_context"] = ""  # 没有 RAG 上下文
    state["plan"] = _build_single_step_plan()
    state["current_step_index"] = 0
    
    mock_decision = RouteDecision(
        next_worker="rag_researcher",
        reasoning="需要检索相关文献"
    )
    
    with patch('src.agents.supervisor.graph.llm') as mock_llm:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_decision
        mock_llm.with_structured_output.return_value = mock_chain
        
        result = supervisor_agent_graph.invoke(state)
        
        assert result.get("next_worker") == "rag_researcher"


def test_supervisor_decision_error_handling(supervisor_state):
    """
    测试 Supervisor 在 LLM 调用失败时的错误处理
    """
    state = supervisor_state.copy()
    state["user_query"] = "测试错误处理"
    state["pending_contribution"] = "待审核内容"
    state["plan"] = _build_single_step_plan()
    state["current_step_index"] = 0
    
    # Mock LLM 抛出异常
    with patch('src.agents.supervisor.graph.llm') as mock_llm:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM API 调用失败")
        mock_llm.with_structured_output.return_value = mock_chain
        
        result = supervisor_agent_graph.invoke(state)
        
        # 断言：当有 pending_contribution 时，应该默认选择 critic
        assert result.get("next_worker") == "critic", \
            "当 LLM 调用失败且有待审核内容时，应默认选择 critic"


def test_supervisor_decision_error_handling_no_pending(supervisor_state):
    """
    测试 Supervisor 在 LLM 调用失败且没有待审核内容时的错误处理
    """
    state = supervisor_state.copy()
    state["user_query"] = "测试错误处理"
    state["pending_contribution"] = None
    state["plan"] = _build_single_step_plan()
    state["current_step_index"] = 0
    
    with patch('src.agents.supervisor.graph.llm') as mock_llm:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM API 调用失败")
        mock_llm.with_structured_output.return_value = mock_chain
        
        result = supervisor_agent_graph.invoke(state)
        
        # 断言：当没有 pending_contribution 时，应该默认选择 rag_researcher
        assert result.get("next_worker") == "rag_researcher", \
            "当 LLM 调用失败且没有待审核内容时，应默认选择 rag_researcher"


def test_supervisor_direct_finish_when_plan_done(supervisor_state):
    """测试计划完成且审核通过时，不依赖 LLM 直接 FINISH。"""
    state = supervisor_state.copy()
    state["plan"] = _build_single_step_plan()
    state["current_step_index"] = 1  # 已超出最后一步
    state["is_approved"] = True
    state["pending_contribution"] = None

    with patch('src.agents.supervisor.graph.llm') as mock_llm:
        result = supervisor_agent_graph.invoke(state)

        assert result.get("next_worker") == "FINISH"
        mock_llm.with_structured_output.assert_not_called()

