from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MISSING = object()


def _get_field(container: Any, field_name: str) -> Any:
    if container is None:
        return _MISSING

    if hasattr(container, field_name):
        return getattr(container, field_name)

    if isinstance(container, Mapping) and field_name in container:
        return container[field_name]

    return _MISSING


def _render_receiver_payload(payload: Any) -> str:
    if payload is None:
        return ""

    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, Mapping):
        data = payload
    else:
        return str(payload)

    sections = {
        "Signals": data.get("signals", []),
        "Evidence": data.get("evidence", []),
        "Implications": data.get("implications", []),
    }
    uncertainty = data.get("uncertainty")
    if uncertainty:
        sections["Uncertainty"] = uncertainty

    lines: list[str] = []
    for title, items in sections.items():
        lines.append(f"### {title}")
        if items:
            lines.extend([f"- {item}" for item in items])
        else:
            lines.append("- None")
        lines.append("")

    return "\n".join(lines).strip()


def extract_expert_output(expert_output: Any, target_field: str) -> str:
    if expert_output is None:
        # Upstream expert node failed (e.g., API overload). Return a safe placeholder so downstream
        # nodes (synthesis/season/weekly planners) can still run and report partial results.
        return _render_receiver_payload(
            {
                "signals": [],
                "evidence": [],
                "implications": [],
                "uncertainty": f"Expert output missing (upstream failure). Cannot extract '{target_field}'.",
            }
        )

    output_container: Any = _MISSING
    if hasattr(expert_output, "output"):
        output_container = expert_output.output
    elif isinstance(expert_output, Mapping):
        output_container = expert_output.get("output")

    if isinstance(output_container, list):
        raise ValueError("Expert output contains questions, not analysis. HITL interaction required.")

    for candidate in (output_container, expert_output):
        payload = _get_field(candidate, target_field)
        if payload is not _MISSING:
            return _render_receiver_payload(payload)

    # If the expert output exists but doesn't contain the expected receiver field, keep the workflow running.
    return _render_receiver_payload(
        {
            "signals": [],
            "evidence": [],
            "implications": [],
            "uncertainty": f"Expert output present but missing '{target_field}' field. Type: {type(expert_output)}",
        }
    )


def extract_agent_content(value: Any) -> str:
    if not value:
        return ""

    if hasattr(value, "output"):
        output = value.output
        if isinstance(output, str):
            return output
        raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")

    if isinstance(value, dict):
        result = value.get("output") or value.get("content")
        if isinstance(result, str):
            return result
        return str(value)

    if isinstance(value, str):
        return value

    return str(value)
