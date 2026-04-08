import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import MetricsExpertOutputs
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

METRICS_SYSTEM_PROMPT_BASE = """## Ziel
Analysiere Trainingsmetriken und die Wettkampfbereitschaft mit datengestützter Präzision.
## Prinzipien
- Analysieren: Konzentriere dich auf Belastungsmuster, Fitnesstrends und Bereitschaft.
- Objektivität: Spekuliere nicht über die Daten hinaus.
- Klarheit: Erkläre komplexe Zusammenhänge einfach.

## Neue Metrik-Definitionen (ACWR V2)

Dir werden "ACWR v2" Metriken zur Verfügung gestellt, die aus der täglichen Trainingsbelastung abgeleitet sind (Summe der activityTrainingLoad pro Tag).

### EWMA Metriken (glatt, reaktionsschnell)
- **Akute EWMA (7 Tage)**: Kurzfristige Belastung (Indikator für Ermüdung).
- **Chronische EWMA (28 Tage)**: Längerfristige Belastung (Indikator für Fitness/Leistungsbereitschaft).
- **Verschobene chronische EWMA (t-7)**: Chronische EWMA vor 7 Tagen bewertet (annähernde Entkopplung).
- **ACWR (EWMA verschoben)**: Akute EWMA / Verschobene chronische EWMA. Verwendung als Indikator für Belastungsspitzen, beachte jedoch, dass die Schwellenwerte eine Kalibrierung erfordern.
- **Risiko-Index**: ln(ACWR) (symmetrisches Maß für „Verdopplung vs. Halbierung“).
- **TSB**: Chronische EWMA - Akute EWMA (negativ = sich ansammelnde Ermüdung).
- **Rampenrate (7 Tage)**: Veränderung der chronischen EWMA im Vergleich zu vor 7 Tagen (erkennt schnellen Belastungsanstieg).
- **Monotonie (7 Tage)**: Durchschnitt(tägliche Belastung der letzten 7 Tage) / Standardabweichung(letzte 7 Tage). Hohe Werte deuten auf geringe Variation hin.
- **Strain (7 Tage)**: (Wöchentliche Gesamtbelastung) x Monotonie.

Note: Schwellenwerte sind Heuristiken und sollten auf den Athleten und die gewählte ACWR-Definition kalibriert werden.

### Rolling-Sum Metriken (Garmin-vergleichbarer Maßstab)
Diese verwenden rollierende 7-Tage-Summen (näher an der Größenordnung von Garmin, obwohl Garmin die Tage möglicherweise anders gewichtet):
- **Akute 7-Tage-Summe**: Summe der täglichen Belastungen der letzten 7 Tage (Garmin-ähnliche akute Größenordnung).
- **Chronischer 28-Tage-Durchschnitt (der akuten 7-Tage-Summe)**: Durchschnitt der letzten 28 Werte der akuten 7-Tage-Summe (geglättete Baseline).
- **ACWR 7d/28d (gekoppelt)**: Akute 7-Tage-Summe / Chronischer 28-Tage-Durchschnitt.
- **ACWR 7d/28d (entkoppelt)**: Akute 7-Tage-Summe / Chronischer 28-Tage-Durchschnitt berechnet bis (t-7), ohne die aktuellste Woche (bevorzugt für Garmin-ähnliches ACWR ohne Kopplung).
"""

METRICS_USER_PROMPT = """## Aufgabe
Analysiere die Zusammenfassung der Metriken, um Muster und Trends zu identifizieren.

## Einschränkungen
- Konzentriere dich auf **globale Trainingsmetriken** (Belastung, VO2max, Status).
- Beschreibe KEINE spezifischen Workouts (Aufgabe des Activity Experts).
- Ziehe KEINE Rückschlüsse auf die interne Physiologie (Aufgabe des Physiology Experts).
- Konzentriere dich darauf, **wie sich der Trainingsreiz im Laufe der Zeit verhält**.

## Inputs
### Metrik-Zusammenfassung
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
- **Kontext**: Dies bildet das **"quantitative Rückgrat"** (Belastungs-/Stressrealität) für den Bericht.
- **Ziel**: Liefere die quantitative Wahrheit der Trainingsbelastung.
- **Freiheit**: Hebe das Belastungsverhalten, Fitnesstrends oder wichtige Verhältnisse hervor.

### 2. `for_season_planner` (12-24 Wochen)
- **Kontext**: Dies informiert die **"Belastungsarchitektur"** (Rampenraten, Volumendeckel) für die Saison.
- **Ziel**: Biete übergeordnete Orientierung zur Kapazität und strukturellen Mustern.
- **Freiheit**: Identifiziere sichere Rampenraten, maximale nachhaltige chronische Belastung oder Volatilitätsgrenzen.

### 3. `for_weekly_planner` (Nächste 28 Tage)
- **Kontext**: Dies dient als **"akute Belastungsschranke"** für die nächsten Wochen.
- **Ziel**: Biete unmittelbare Belastungsführung und Grenzwerte.
- **Freiheit**: Definiere Sicherheitsgrenzen, Push/Pull-Signale oder spezifische Belastungsziele.
- **KRITISCH**: Verordne KEINE spezifischen Workouts. Gib Grenzwerte und Belastungshinweise.
"""

METRICS_FINAL_CHECKLIST = """
## Abschluss-Checkliste
- Verwende Signale/Beweise/Auswirkungen/Unsicherheit pro Empfänger.
- Bleibe ausschließlich im Bereich der Metriken.
- Keine Verordnungen für spezifische Workouts.
"""

async def metrics_expert_node(state: TrainingAnalysisState) -> dict[str, list | str | dict]:
    logger.info("Starting metrics expert analysis node")

    plot_storage = PlotStorage(state["execution_id"])
    plotting_enabled = state.get("plotting_enabled", False)
    hitl_enabled = state.get("hitl_enabled", True)

    logger.info(
        "Metrics expert: Plotting %s, HITL %s",
        "enabled" if plotting_enabled else "disabled",
        "enabled" if hitl_enabled else "disabled",
    )

    tools = configure_node_tools(
        agent_name="metrics",
        plot_storage=plot_storage,
        plotting_enabled=plotting_enabled,
    )

    system_prompt = (
        get_workflow_context("metrics")
        + METRICS_SYSTEM_PROMPT_BASE
        + (get_plotting_instructions("metrics") if plotting_enabled else "")
        + (get_hitl_instructions("metrics") if hitl_enabled else "")
        + METRICS_FINAL_CHECKLIST
    )

    base_llm = ModelSelector.get_llm(AgentRole.METRICS_EXPERT)

    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(MetricsExpertOutputs)

    agent_start_time = datetime.now()

    async def call_metrics_with_tools():
        qa_messages = normalize_langchain_messages(state.get("metrics_expert_messages", []))

        base_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": METRICS_USER_PROMPT.format(
                    data=state.get("metrics_summary", "No metrics summary available"),
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
        result = await retry_with_backoff(
            call_metrics_with_tools, AI_ANALYSIS_CONFIG, "Metrics Agent with Tools"
        )
        if isinstance(result, tuple) and len(result) == 2:
            agent_output, usage = result
        else:
            agent_output, usage = result, {}
        logger.info("Metrics expert analysis completed")

        execution_time = (datetime.now() - agent_start_time).total_seconds()

        plots, plot_storage_data, available_plots = create_plot_entries("metrics", plot_storage)

        log_node_completion("Metrics expert analysis", execution_time, len(available_plots))

        return {
            "metrics_outputs": agent_output,
            "plots": plots,
            "plot_storage_data": plot_storage_data,
            "costs": [create_cost_entry("metrics", execution_time)],
            "usage_metadata": extract_usage_metadata(agent_output, AgentRole.METRICS_EXPERT, usage),
            "available_plots": available_plots,
        }

    return await execute_node_with_error_handling(
        node_name="Metrics expert analysis",
        node_function=node_execution,
        error_message_prefix="Metrics expert analysis failed",
    )
