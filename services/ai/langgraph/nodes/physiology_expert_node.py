import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import PhysiologyExpertOutputs
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

PHYSIOLOGY_SYSTEM_PROMPT_BASE = """## Ziel
Optimiere Erholung und Anpassung durch präzise physiologische Analyse.
## Prinzipien
- Ganzheitlich: Betrachte den Körper als ein vernetztes System.
- Zeitlich: Interpretiere Signale über unmittelbare und langfristige Zeiträume hinweg.
- Handlungsorientiert: Identifiziere Erholungsfenster und Belastungskosten."""

PHYSIOLOGY_USER_PROMPT = """## Aufgabe
Analysiere die physiologische Zusammenfassung, um Erholung und Anpassung zu bewerten.

## Einschränkungen
- Konzentriere dich auf den **internen Zustand** (HRV, Schlaf, Ruhepuls, Stress).
- Leite KEINE Belastungsmetriken neu ab (Aufgabe des Metrics Experts).
- Entwirf KEINE Trainingsstruktur neu (Aufgabe des Planners).
- Konzentriere dich darauf, **wie der Körper mit Stress umgeht**.

## Inputs
### Physiologische Zusammenfassung
{data}
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
- **Kontext**: Dies speist die Sicht auf den **"ganzen Athleten"** (Zusammenfassung & Synthese). Es dient als "interner Körper-Check".
- **Ziel**: Biete eine qualitative Bewertung von Erholung und Anpassung.
- **Freiheit**: Hebe Erholungskosten, Anpassungsstatus oder interne Signale hervor.

### 2. `for_season_planner` (12-24 Wochen)
- **Kontext**: Dies informiert **langfristige strukturelle Entscheidungen** (Makrozyklus).
- **Ziel**: Informiere über die **"Absorptionskapazität"** des Athleten.
- **Freiheit**: Konzentriere dich auf langfristige Robustheit, Crash-Risiken und Resilienz.

### 3. `for_weekly_planner` (Nächste 28 Tage)
- **Kontext**: Dies fungiert als **"Ampel"** (Bereitschaftsbegrenzer) für den nächsten Block.
- **Ziel**: Biete Orientierung zur Bereitschaft.
- **Freiheit**: Sprich in **Bereitschaftskorridoren** (z. B. "Hohe Bereitschaft, Overload möglich" oder "Sympathische Dominanz, Intensität begrenzen")."""

PHYSIOLOGY_FINAL_CHECKLIST = """
## Abschluss-Checkliste
- Verwende Signale/Beweise/Auswirkungen/Unsicherheit pro Empfänger.
- Bleibe ausschließlich im Bereich der Physiologie.
- Keine Neugestaltung der Trainingsstruktur.
"""

async def physiology_expert_node(state: TrainingAnalysisState) -> dict[str, list | str | dict]:
    logger.info("Starting physiology expert analysis node")

    plot_storage = PlotStorage(state["execution_id"])
    plotting_enabled = state.get("plotting_enabled", False)
    hitl_enabled = state.get("hitl_enabled", True)

    logger.info(
        "Physiology expert: Plotting %s, HITL %s",
        "enabled" if plotting_enabled else "disabled",
        "enabled" if hitl_enabled else "disabled",
    )

    tools = configure_node_tools(
        agent_name="physiology",
        plot_storage=plot_storage,
        plotting_enabled=plotting_enabled,
    )

    system_prompt = (
        get_workflow_context("physiology")
        + PHYSIOLOGY_SYSTEM_PROMPT_BASE
        + (get_plotting_instructions("physiology") if plotting_enabled else "")
        + (get_hitl_instructions("physiology") if hitl_enabled else "")
        + PHYSIOLOGY_FINAL_CHECKLIST
    )

    base_llm = ModelSelector.get_llm(AgentRole.PHYSIOLOGY_EXPERT)
    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(PhysiologyExpertOutputs)

    agent_start_time = datetime.now()

    async def call_physiology_analysis():
        qa_messages = normalize_langchain_messages(state.get("physiology_expert_messages", []))

        base_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": PHYSIOLOGY_USER_PROMPT.format(
                    data=state.get("physiology_summary", "No physiology summary available"),
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
            call_physiology_with_tools, AI_ANALYSIS_CONFIG, "Physiology Agent with Tools"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        plots, plot_storage_data, available_plots = create_plot_entries("physiology", plot_storage)

        log_node_completion("Physiology expert analysis", execution_time, len(available_plots))

        return {
            "physiology_outputs": agent_output,
            "plots": plots,
            "plot_storage_data": plot_storage_data,
            "costs": [create_cost_entry("physiology", execution_time)],
            "usage_metadata": extract_usage_metadata(agent_output, AgentRole.PHYSIOLOGY_EXPERT, usage),
            "available_plots": available_plots,
        }

    return await execute_node_with_error_handling(
        node_name="Physiology expert analysis",
        node_function=node_execution,
        error_message_prefix="Physiology expert analysis failed",
    )
