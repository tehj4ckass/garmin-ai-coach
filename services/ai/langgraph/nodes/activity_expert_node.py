import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import ActivityExpertOutputs
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.message_helper import normalize_langchain_messages
from services.ai.model_config import ModelSelector
from services.ai.tools.plotting import PlotStorage
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import (
    configure_node_tools,
    create_cost_entry,
    create_plot_entries,
    execute_node_with_error_handling,
    extract_usage_metadata,
    log_node_completion,
)
from .prompt_components import (
    get_hitl_instructions,
    get_plotting_instructions,
    get_workflow_context,
)
from .tool_calling_helper import handle_tool_calling_in_node

logger = logging.getLogger(__name__)

ACTIVITY_EXPERT_SYSTEM_PROMPT_BASE = """## Ziel
Interpretiere strukturierte Aktivitätsdaten, um Muster in der Workout-Progression zu optimieren.
## Prinzipien
- Präzision: Erkenne subtile Details in der Ausführung.
- Mustererkennung: Identifiziere, was funktioniert und was nicht.
- Klarheit: Durchbrich Verwirrung mit direkter Analyse."""

ACTIVITY_EXPERT_USER_PROMPT = """## Aufgabe
Interpretiere Aktivitätszusammenfassungen, um Muster und Orientierungshilfen zu identifizieren.

## Einschränkungen
- Konzentriere dich auf die **Ausführung auf Sitzungsebene** (Pace, Leistung, HF, Struktur).
- Erkläre KEINE globale Belastung (Aufgabe des Metrics Experts).
- Schlage KEINE zukünftigen Zeitpläne vor (Aufgabe des Planners).
- Konzentriere dich darauf, **"was dieses spezifische Workout mit dem System macht"**.

## Inputs
### Aktivitätszusammenfassung
{activity_summary}
### Kontext
- Wettkämpfe: ```json {competitions} ```
- Datum: ```json {current_date} ```
- **Benutzer-Kontext**: ``` {analysis_context} ```

## Ausgabeanforderungen
Erstelle 3 strukturierte Felder. Verwende für JEDES Feld dieses interne Layout:
- **Signale (Signals)**: was hat sich geändert (prägnant)
- **Beweise (Evidence)**: Zahlen + Datumsbereiche
- **Auswirkungen (Implications)**: Einschränkungen/Möglichkeiten für diesen Empfänger
- **Unsicherheit (Uncertainty)**: Lücken/geringe Abdeckung, falls vorhanden

**Wichtig**: Schneide den Inhalt auf den jeweiligen Empfänger zu.

### 1. `for_synthesis` (Umfassender Bericht)
- **Kontext**: Dies speist die Sicht auf den **"ganzen Athleten"** (Zusammenfassung & Synthese).
- **Ziel**: Biete eine qualitative Bewertung der Ausführungsqualität.
- **Freiheit**: Hebe hervor, was am wichtigsten ist – Ausführungsmuster, Progressionsqualität oder Konsistenz.

### 2. `for_season_planner` (12-24 Wochen)
- **Kontext**: Dies informiert **langfristige strukturelle Entscheidungen** (Makrozyklus).
- **Ziel**: Identifiziere, welche "Bausteine" (Workout-Typen) für diesen spezifischen Athleten effektiv sind.
- **Freiheit**: Konzentriere dich auf Erfolgsmuster und Sequenzierungspräferenzen.

### 3. `for_weekly_planner` (Nächste 28 Tage)
- **Kontext**: Dies informiert die **unmittelbare Planung & Einschränkungen** (Mesozyklus).
- **Ziel**: Liefere umsetzbare Regeln für den nächsten Block.
- **Freiheit**: Definiere Einschränkungen, Möglichkeiten und Hinweise zur Sitzungsbelastung nach Bedarf.
- **KRITISCH**: Schlage KEINEN Zeitplan vor. Liefere Regeln und Bausteine."""

ACTIVITY_FINAL_CHECKLIST = """
## Abschluss-Checkliste
- Verwende Signale/Beweise/Auswirkungen/Unsicherheit pro Empfänger.
- Bleibe ausschließlich im Bereich der Aktivitätsausführung.
- Keine Zeitplanschläge.
"""


async def activity_expert_node(state: TrainingAnalysisState) -> dict[str, list | str | dict]:
    logger.info("Starting activity expert node")

    plot_storage = PlotStorage(state["execution_id"])
    plotting_enabled = state.get("plotting_enabled", False)
    hitl_enabled = state.get("hitl_enabled", True)

    logger.info(
        "Activity expert node: Plotting %s, HITL %s",
        "enabled" if plotting_enabled else "disabled",
        "enabled" if hitl_enabled else "disabled",
    )

    tools = configure_node_tools(
        agent_name="activity",
        plot_storage=plot_storage,
        plotting_enabled=plotting_enabled,
    )

    system_prompt = (
        get_workflow_context("activity")
        + ACTIVITY_EXPERT_SYSTEM_PROMPT_BASE
        + (get_plotting_instructions("activity") if plotting_enabled else "")
        + (get_hitl_instructions("activity") if hitl_enabled else "")
        + ACTIVITY_FINAL_CHECKLIST
    )

    base_llm = ModelSelector.get_llm(AgentRole.ACTIVITY_EXPERT)
    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(ActivityExpertOutputs)

    agent_start_time = datetime.now()

    async def call_activity_expert():
        qa_messages = normalize_langchain_messages(state.get("activity_expert_messages", []))

        base_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": ACTIVITY_EXPERT_USER_PROMPT.format(
                    activity_summary=state.get("activity_summary", ""),
                    competitions=json.dumps(state["competitions"], indent=2),
                    current_date=json.dumps(state["current_date"], indent=2),
                    analysis_context=state["analysis_context"],
                ),
            },
        ]

        return await handle_tool_calling_in_node(
            llm_with_tools=llm_with_structure,
            messages=base_messages + qa_messages,
            tools=tools,
            max_iterations=15,
        )

    async def node_execution():
        agent_output, usage = await retry_with_backoff(
            call_activity_expert, AI_ANALYSIS_CONFIG, "Activity Expert Analysis with Tools"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        plots, plot_storage_data, available_plots = create_plot_entries("activity_expert", plot_storage)

        log_node_completion("Activity expert", execution_time, len(available_plots))

        return {
            "activity_outputs": agent_output,
            "plots": plots,
            "plot_storage_data": plot_storage_data,
            "costs": [create_cost_entry("activity_expert", execution_time)],
            "usage_metadata": extract_usage_metadata(agent_output, AgentRole.ACTIVITY_EXPERT, usage),
            "available_plots": available_plots,
        }

    return await execute_node_with_error_handling(
        node_name="Activity expert",
        node_function=node_execution,
        error_message_prefix="Activity expert analysis failed",
    )
