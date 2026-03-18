import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import AgentOutput
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.output_helper import extract_expert_output
from services.ai.model_config import ModelSelector
from services.ai.utils.plan_storage import FilePlanStorage
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import (
    configure_node_tools,
    create_cost_entry,
    execute_node_with_error_handling,
    extract_usage_metadata,
    log_node_completion,
)
from .prompt_components import get_hitl_instructions, get_workflow_context
from .tool_calling_helper import handle_tool_calling_in_node

logger = logging.getLogger(__name__)

SEASON_PLANNER_SYSTEM_PROMPT = """Du bist ein strategischer Saisonplaner.
## Ziel
Erstelle strategische Saisonpläne für die langfristige athletische Entwicklung.
## Prinzipien
- Strategisch: Konzentriere dich auf Makrozyklen und Phasen.
- Adaptiv: Nutze Experten-Erkenntnisse, um den Plan maßzuschneideren.
- Systematisch: Stelle eine logische Progression hin zu den Zielen sicher."""

SEASON_PLANNER_USER_PROMPT = """Erstelle einen STRATEGISCHEN, ÜBERGEORDNETEN Saisonplan (12-24 Wochen).

## Inputs
- Athlet: {athlete_name}
- Datum: ```json {current_date} ```
- Wettkämpfe: ```json {competitions} ```

## Experten-Erkenntnisse
### Metriken
```markdown
{metrics_insights}
```
### Aktivität
```markdown
{activity_insights}
```
### Physiologie
```markdown
{physiology_insights}
```

## Aufgabe
Erstelle ein Makrozyklus-Gerüst.
- **Integrieren**: Nutze Experten-Erkenntnisse als deinen Polarstern.
- **Strategisch planen**: Definiere Phasen, Themen und Fokusbereiche.
- **Grenzen respektieren**: Verordne KEINE täglichen Workouts (Aufgabe des Weekly Planners).

## Ausgabeanforderungen
Formatierung als strukturiertes Markdown.
1. **Phasen**: Definiere Phasen (Basis, Aufbau etc.) mit Zielen und Themen.
2. **Experten-Begründung**: Referenziere explizit, wie Metriken, Aktivität und Physiologie den Plan beeinflusst haben.
3. **Einschränkungen**: Qualitative Einschränkungen, die von den Experten abgeleitet wurden.

**Bleibe auf hoher Ebene**. Entwirf die **Karte der Saison**, nicht die Abbiegehinweise. **FASSE DICH KURZ**."""




async def season_planner_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting season planner node")

    hitl_enabled = state.get("hitl_enabled", True)
    logger.info("Season planner node: HITL %s", "enabled" if hitl_enabled else "disabled")

    agent_start_time = datetime.now()

    tools = configure_node_tools(
        agent_name="season_planner",
        plot_storage=None,
        plotting_enabled=False,
    )

    system_prompt = (
        SEASON_PLANNER_SYSTEM_PROMPT +
        get_workflow_context("season_planner") +
        (get_hitl_instructions("season_planner") if hitl_enabled else "")
    )

    qa_messages_raw = state.get("season_planner_messages", [])
    qa_messages = []
    for msg in qa_messages_raw:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
            qa_messages.append({"role": role, "content": msg.content})
        else:
            qa_messages.append(msg)

    existing_season_plan = ""
    try:
        storage = FilePlanStorage()
        loaded_plan = storage.load_plan(state["user_id"], "season_plan")
        if loaded_plan:
            existing_season_plan = loaded_plan
    except Exception as exc:
        logger.warning("Could not read existing season plan: %s", exc)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": SEASON_PLANNER_USER_PROMPT.format(
            athlete_name=state["athlete_name"],
            current_date=json.dumps(state["current_date"], indent=2),
            competitions=json.dumps(state["competitions"], indent=2),
            metrics_insights=extract_expert_output(state.get("metrics_outputs"), "for_season_planner"),
            activity_insights=extract_expert_output(state.get("activity_outputs"), "for_season_planner"),
            physiology_insights=extract_expert_output(state.get("physiology_outputs"), "for_season_planner"),
        ) + (f"\n\n## Bestehender Saisonplan\nWir haben einen bestehenden Saisonplan. Beginne NICHT bei Null. Überprüfe diesen Plan anhand der neuen Experten-Erkenntnisse. Wenn der Plan immer noch gültig ist, behalte die Phasenstruktur bei und verfeinere nur die Details. Triggere eine vollständige Neuplanung nur dann, wenn die neuen Daten darauf hindeuten, dass der alte Plan gefährlich vom Kurs abgekommen ist.\n\n```markdown\n{existing_season_plan}\n```" if existing_season_plan else "")},
    ]

    base_llm = ModelSelector.get_llm(AgentRole.SEASON_PLANNER)

    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(AgentOutput)

    async def call_season_planning():
        messages_with_qa = base_messages + qa_messages
        if tools:
            return await handle_tool_calling_in_node(
                llm_with_tools=llm_with_structure,
                messages=messages_with_qa,
                tools=tools,
                max_iterations=15,
            )
        else:
            return await llm_with_structure.ainvoke(messages_with_qa)

    async def node_execution():
        planning_result = await retry_with_backoff(
            call_season_planning, AI_ANALYSIS_CONFIG, "Season Planning"
        )

        # Handle potential tuple return from handle_tool_calling_in_node
        if isinstance(planning_result, tuple):
            agent_output, usage = planning_result
        else:
            agent_output = planning_result
            usage = None

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        log_node_completion("Season planning", execution_time)

        return {
            "season_plan": agent_output.model_dump(),
            "costs": [create_cost_entry("season_planner", execution_time)],
            "usage_metadata": extract_usage_metadata(agent_output, AgentRole.SEASON_PLANNER, usage),
        }

    return await execute_node_with_error_handling(
        node_name="Season planner",
        node_function=node_execution,
        error_message_prefix="Season planning failed",
    )
