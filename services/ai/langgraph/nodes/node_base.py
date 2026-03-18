import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langgraph.errors import GraphInterrupt
from services.ai.tools.plotting import PlotStorage, create_plotting_tools

from services.ai.ai_settings import AgentRole, ai_settings


logger = logging.getLogger(__name__)


def extract_usage_metadata(response: Any, role: AgentRole, usage_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    model_name = ai_settings.get_model_for_role(role)
    usage_metadata = {}

    if usage_dict:
        usage_metadata[model_name] = usage_dict
        return usage_metadata

    # Standard LangChain usage_metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        usage_metadata[model_name] = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)),
        }

    return usage_metadata


def configure_node_tools(
    agent_name: str,
    plot_storage: PlotStorage | None = None,
    plotting_enabled: bool = False,
) -> list:
    tools = []

    if plotting_enabled and plot_storage:
        plotting_tool = create_plotting_tools(plot_storage, agent_name=agent_name)
        tools.append(plotting_tool)
        logger.debug("%s: Added plotting tool", agent_name)

    return tools


def create_cost_entry(agent_name: str, execution_time: float) -> dict[str, Any]:

    return {
        "agent": agent_name,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat(),
    }


def create_plot_entries(agent_name: str, plot_storage: PlotStorage) -> tuple[list, dict, list]:

    all_plots = plot_storage.get_all_plots()
    timestamp_iso = datetime.now().isoformat()

    plots = [
        {
            "agent": agent_name,
            "plot_id": plot_id,
            "timestamp": timestamp_iso,
        }
        for plot_id in all_plots
    ]

    plot_storage_data = {
        plot_id: {
            "plot_id": metadata.plot_id,
            "description": metadata.description,
            "agent_name": metadata.agent_name,
            "created_at": metadata.created_at.isoformat(),
            "html_content": metadata.html_content,
            "data_summary": metadata.data_summary,
        }
        for plot_id, metadata in all_plots.items()
    }

    return plots, plot_storage_data, list(all_plots.keys())


async def execute_node_with_error_handling(
    node_name: str,
    node_function: Callable,
    error_message_prefix: str,
) -> dict[str, Any]:

    try:
        return await node_function()

    except GraphInterrupt:
        logger.info("%s: GraphInterrupt raised - propagating to LangGraph", node_name)
        raise

    except Exception as exc:
        logger.exception("%s failed", node_name)
        return {"errors": [f"{error_message_prefix}: {exc!s}"]}


def log_node_completion(node_name: str, execution_time: float, plot_count: int = 0) -> None:
    if plot_count > 0:
        logger.info("%s completed in %.2fs with %s plots", node_name, execution_time, plot_count)
        return

    logger.info("%s completed in %.2fs", node_name, execution_time)
