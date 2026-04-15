import pytest

from services.ai.langgraph.nodes.prompt_components import format_valid_plot_catalog
from services.ai.tools.plotting.langgraph_plotting_tool import create_plotting_tools
from services.ai.tools.plotting.plot_storage import PlotMetadata, PlotStorage
from services.ai.tools.plotting.reference_resolver import repair_misnamed_plot_references


class TestPlottingToolIntegration:
    def test_format_valid_plot_catalog_lists_ids(self):
        data = {
            "metrics_1_001": {
                "plot_id": "metrics_1_001",
                "description": "ACWR",
                "agent_name": "metrics",
                "created_at": "2026-01-01T12:00:00",
                "html_content": "<div/>",
                "data_summary": "",
            }
        }
        text = format_valid_plot_catalog(data)
        assert "`[PLOT:metrics_1_001]`" in text
        assert "ACWR" in text

    def test_repair_misnamed_plot_references_single_plot(self):
        storage = PlotStorage("exec")
        storage.plots["metrics_99_001"] = PlotMetadata(
            plot_id="metrics_99_001",
            description="x",
            agent_name="metrics",
            created_at=__import__("datetime").datetime.now(),
            html_content="<p/>",
            data_summary="",
        )
        raw = "See [PLOT:load_acwr_tsb] here."
        fixed, n = repair_misnamed_plot_references(raw, storage)
        assert n == 1
        assert "[PLOT:metrics_99_001]" in fixed
        assert "load_acwr_tsb" not in fixed

    def test_repair_skipped_when_multiple_plots(self):
        storage = PlotStorage("exec")
        now = __import__("datetime").datetime.now()
        for pid in ("a_1_001", "b_2_002"):
            storage.plots[pid] = PlotMetadata(
                plot_id=pid,
                description="x",
                agent_name="m",
                created_at=now,
                html_content="<p/>",
                data_summary="",
            )
        raw = "[PLOT:wrong]"
        fixed, n = repair_misnamed_plot_references(raw, storage)
        assert n == 0
        assert fixed == raw



    def test_langchain_tool_creation(self):
        plot_storage = PlotStorage("test_execution")

        plotting_tool = create_plotting_tools(plot_storage, agent_name="test")

        assert plotting_tool.name == "python_plotting_tool"

        assert "Execute complete Python code" in plotting_tool.description

    @pytest.mark.asyncio
    async def test_tool_invocation(self):
        plot_storage = PlotStorage("test_execution")
        plotting_tool = create_plotting_tools(plot_storage, agent_name="test")

        test_code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='Test Data'))
fig.update_layout(title='Test Plot')
"""

        result = await plotting_tool.ainvoke(
            {"python_code": test_code, "description": "Test plot for integration"}
        )

        assert result["ok"] is True
        assert "Plot created successfully" in result["message"]
        assert "[PLOT:" in result["message"]
        assert "plot_id" in result

    @pytest.mark.asyncio
    async def test_model_tool_binding(self, monkeypatch):
        from unittest.mock import Mock

        from services.ai.ai_settings import AgentRole
        from services.ai.model_config import ModelSelector

        plot_storage = PlotStorage("test_execution")
        plotting_tool = create_plotting_tools(plot_storage, agent_name="test")

        mock_llm = Mock()
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.kwargs = {"tools": [plotting_tool]}
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        monkeypatch.setattr(ModelSelector, "get_llm", lambda role: mock_llm)

        llm = ModelSelector.get_llm(AgentRole.METRICS_EXPERT)
        llm_with_tools = llm.bind_tools([plotting_tool])

        assert hasattr(llm_with_tools, "kwargs") and "tools" in llm_with_tools.kwargs
        assert len(llm_with_tools.kwargs["tools"]) == 1
        assert plotting_tool.name == "python_plotting_tool"

    def test_tools_condition_compatibility(self):
        from langchain_core.messages import AIMessage
        from langgraph.prebuilt import tools_condition

        message_without_tools = AIMessage(content="This is a regular response")

        result = tools_condition({"messages": [message_without_tools]})
        assert result == "__end__"

        message_with_tools = AIMessage(
            content="I'll create a plot for you",
            tool_calls=[
                {
                    "name": "python_plotting_tool",
                    "args": {"python_code": "test", "description": "test"},
                    "id": "test_id",
                }
            ],
        )

        result = tools_condition({"messages": [message_with_tools]})
        assert result == "tools"

    def test_canonical_pattern_components(self):
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import ToolNode

        plot_storage = PlotStorage("test_execution")
        plotting_tool = create_plotting_tools(plot_storage, "test")
        tools = [plotting_tool]

        tool_node = ToolNode(tools)
        assert tool_node is not None

        from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState

        workflow = StateGraph(TrainingAnalysisState)
        assert workflow is not None


if __name__ == "__main__":
    pytest.main([__file__])
