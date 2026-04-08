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

    # Expert outputs can be either questions OR structured receiver outputs.
    output_container: Any = _MISSING
    if hasattr(expert_output, "outputs") or hasattr(expert_output, "questions"):
        questions = getattr(expert_output, "questions", None)
        outputs = getattr(expert_output, "outputs", None)
        if questions:
            raise ValueError("Expert output contains questions, not analysis. HITL interaction required.")
        output_container = outputs
    elif hasattr(expert_output, "output"):
        # Backwards compatibility (older persisted artifacts / dict-like fallbacks)
        output_container = expert_output.output
        if isinstance(output_container, list):
            raise ValueError("Expert output contains questions, not analysis. HITL interaction required.")
    elif isinstance(expert_output, Mapping):
        # Dict-like fallback
        questions = expert_output.get("questions")
        outputs = expert_output.get("outputs")
        if questions:
            raise ValueError("Expert output contains questions, not analysis. HITL interaction required.")
        output_container = outputs if outputs is not None else expert_output.get("output")
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

    if hasattr(value, "content") or hasattr(value, "questions"):
        questions = getattr(value, "questions", None)
        if questions:
            raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")
        content = getattr(value, "content", "")
        return content or ""

    if hasattr(value, "output"):
        # Backwards compatibility
        output = value.output
        if isinstance(output, str):
            return output
        raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")

    if isinstance(value, dict):
        if value.get("questions"):
            raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")
        result = value.get("content") or value.get("output")
        if isinstance(result, str):
            return result
        return str(value)

    if isinstance(value, str):
        return value

    return str(value)
