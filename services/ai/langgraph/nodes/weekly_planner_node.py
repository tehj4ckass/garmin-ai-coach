import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import AgentOutput
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.message_helper import normalize_langchain_messages
from services.ai.langgraph.utils.output_helper import extract_agent_content, extract_expert_output
from services.ai.model_config import ModelSelector
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

WEEKLY_PLANNER_SYSTEM_PROMPT = """## Ziel
Erstelle detaillierte, praktische Trainingspläne, die Belastung und Erholung ausbalancieren.
## Prinzipien
- Anpassung: Progressive Überlastung mit angemessener Erholung.
- Spezifität: Das Training muss den Anforderungen des Events entsprechen.
- Individualisierung: Passe das Training an den aktuellen Zustand und die Geschichte des Athleten an."""

WEEKLY_PLANNER_USER_PROMPT = """## Aufgabe
Erstelle einen detaillierten 28-Tage-Trainingsplan (4 Wochen).

## Einschränkungen
- **Phase berücksichtigen**: Priorisiere die Phasen-Absicht des Saisonplans.
- **Bereitschaft respektieren**: Passe die Intensität basierend auf Physiologie-/Metrik-Signalen an (z. B. Belastung reduzieren, wenn die Erholung niedrig ist).
- **Signale integrieren**: Nutze die Ratschläge des Activity Experts für die Sitzungsstruktur.
- **Kürze**: Verwende Standardnotation (z. B. "4x(5' Z4, 2' r)"), um den Plan kompakt zu halten.

## Inputs
### Saisonplan
```markdown
{season_plan}
```
### Athleten-Kontext
- Name: {athlete_name}
- Datum: ```json {current_date} ```
- Kommende Wochen: ```json {week_dates} ```
- Wettkämpfe: ```json {competitions} ```
- **Benutzer-Kontext**: ``` {planning_context} ```

### Expertenanalyse
- Metriken: ``` {metrics_analysis} ```
- Aktivität: ``` {activity_analysis} ```
- Physiologie: ``` {physiology_analysis} ```

## Ausgabeanforderungen
1. **Zonen-Tabelle**: Definiere zuerst die Intensitätszonen.
2. **Struktur**: Gruppiere nach Woche (1-4).
3. **Tägliches Format**:
   - **TAG & DATUM**: z. B. "Mo, 24. Nov"
   - **FOKUS**: 1-2 Wörter (z. B. "Erholung", "VO2max")
   - **WORKOUT**: Kompakter Struktur-String.
   - **ZWECK**: Ein kurzer Satz.
   - **ANPASSUNG**: "Bei Müdigkeit: ..."

**Wichtig:**
- Nutze aktuelle Aktivitätsdaten, um den aktuellen Trainingsfluss fortzusetzen und beginne keine neue Phase.
- Nutze den Saisonplan als Leitfaden, aber erzwinge ihn nicht.
- Platziere die Sitzungen klug, um aufeinanderfolgende hochintensive Sitzungen oder Krafttrainingseinheiten etc. zu vermeiden.
"""

WEEKLY_PLANNER_FINAL_CHECKLIST = """
## Abschluss-Checkliste
- Beachte den 28-Tage-Horizont und die Wochengruppierung.
- Widerspreche nicht den Experten-Einschränkungen.
- Halte die Ausgabe kompakt und strukturiert.
"""


async def weekly_planner_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting weekly planner node")

    hitl_enabled = state.get("hitl_enabled", True)
    logger.info("Weekly planner node: HITL %s", "enabled" if hitl_enabled else "disabled")

    agent_start_time = datetime.now()

    tools = configure_node_tools(
        agent_name="weekly_planner",
        plot_storage=None,
        plotting_enabled=False,
    )

    system_prompt = (
        get_workflow_context("weekly_planner")
        + WEEKLY_PLANNER_SYSTEM_PROMPT
        + (get_hitl_instructions("weekly_planner") if hitl_enabled else "")
        + WEEKLY_PLANNER_FINAL_CHECKLIST
    )

    qa_messages = normalize_langchain_messages(state.get("weekly_planner_messages", []))
    user_message = {
        "role": "user",
        "content": WEEKLY_PLANNER_USER_PROMPT.format(
            season_plan=extract_agent_content(state.get("season_plan")),
            athlete_name=state["athlete_name"],
            current_date=json.dumps(state["current_date"], indent=2),
            week_dates=json.dumps(state["week_dates"], indent=2),
            competitions=json.dumps(state["competitions"], indent=2),
            planning_context=state["planning_context"],
            metrics_analysis=extract_expert_output(state.get("metrics_outputs"), "for_weekly_planner"),
            activity_analysis=extract_expert_output(state.get("activity_outputs"), "for_weekly_planner"),
            physiology_analysis=extract_expert_output(state.get("physiology_outputs"), "for_weekly_planner"),
        ),
    }
    base_messages = [{"role": "system", "content": system_prompt}, user_message]

    base_llm = ModelSelector.get_llm(AgentRole.WORKOUT)
    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(AgentOutput)

    async def call_weekly_planning():
        messages_with_qa = base_messages + qa_messages
        if tools:
            return await handle_tool_calling_in_node(
                llm_with_tools=llm_with_structure,
                messages=messages_with_qa,
                tools=tools,
                max_iterations=15,
            )
        return await llm_with_structure.ainvoke(messages_with_qa)

    async def node_execution():
        planning_result = await retry_with_backoff(
            call_weekly_planning, AI_ANALYSIS_CONFIG, "Weekly Planning"
        )

        # Handle potential tuple return from handle_tool_calling_in_node
        if isinstance(planning_result, tuple):
            agent_output, usage = planning_result
        else:
            agent_output = planning_result
            usage = None

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        log_node_completion("Weekly planning", execution_time)

        return {
            "weekly_plan": agent_output.model_dump(),
            "costs": [create_cost_entry("weekly_planner", execution_time)],
            "usage_metadata": extract_usage_metadata(agent_output, AgentRole.WORKOUT, usage),
        }

    return await execute_node_with_error_handling(
        node_name="Weekly planner",
        node_function=node_execution,
        error_message_prefix="Weekly planning failed",
    )
