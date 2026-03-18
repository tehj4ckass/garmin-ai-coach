import logging
from typing import Any

from langchain_core.messages import ToolMessage

from langgraph.errors import GraphInterrupt

logger = logging.getLogger(__name__)


def extract_text_content(response) -> str:
    if hasattr(response, "content_blocks"):
        try:
            blocks = response.content_blocks
            text_parts = [b["text"] for b in blocks if b.get("type") == "text" and "text" in b]
            if text_parts:
                return "".join(text_parts)
        except Exception as e:
            logger.debug("Failed to extract from content_blocks: %s", e)

    content = response.content if hasattr(response, "content") else response

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_item = next(
            (item["text"] for item in content
             if isinstance(item, dict) and item.get("type") == "text" and "text" in item),
            None
        )
        if text_item:
            return text_item

        text_item = next(
            (item["text"] for item in content
             if isinstance(item, dict) and "text" in item),
            None
        )
        if text_item:
            return text_item

    return str(content)


async def handle_tool_calling_in_node(
    llm_with_tools, messages: list[dict[str, str]], tools: list, max_iterations: int = 5
):
    conversation: list[Any] = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages if msg["role"] in ("system", "user", "assistant")
    ]

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        logger.debug("Tool calling iteration %s", iteration)

        response = await llm_with_tools.ainvoke(conversation)

        # Accumulate usage if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            total_usage["input_tokens"] += getattr(usage, "input_tokens", 0)
            total_usage["output_tokens"] += getattr(usage, "output_tokens", 0)
            total_usage["total_tokens"] += getattr(usage, "total_tokens", getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))

        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info("LLM requested %s tool calls", len(response.tool_calls))

            conversation.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                logger.info("Executing tool: %s", tool_name)

                tool = next(
                    (tool_candidate for tool_candidate in tools
                     if hasattr(tool_candidate, "name") and tool_candidate.name == tool_name),
                    None
                )

                if tool is None:
                    error_msg = f"Tool {tool_name} not found"
                    logger.error(error_msg)
                    tool_result = error_msg
                else:
                    try:
                        if hasattr(tool, "ainvoke"):
                            tool_result = await tool.ainvoke(tool_args)
                        elif hasattr(tool, "invoke"):
                            tool_result = tool.invoke(tool_args)
                        elif callable(tool):
                            tool_result = await tool.ainvoke(tool_args)
                        else:
                            tool_result = f"Unable to invoke tool {tool_name}"

                        logger.info("Tool %s executed successfully", tool_name)

                    except GraphInterrupt:
                        logger.info("Tool %s triggered HITL interrupt - pausing workflow", tool_name)
                        raise

                tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                conversation.append(tool_message)

        else:
            logger.info("Final response received after %s iterations", iteration)
            return response, total_usage

    logger.warning("Max iterations (%s) reached in tool calling", max_iterations)
    return response, total_usage
