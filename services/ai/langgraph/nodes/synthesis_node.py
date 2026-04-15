import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.output_helper import extract_expert_output
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import extract_usage_metadata
from .prompt_components import format_valid_plot_catalog
from .tool_calling_helper import handle_tool_calling_in_node

logger = logging.getLogger(__name__)





SYNTHESIS_SYSTEM_PROMPT_BASE = """Du bist ein Spezialist für Leistungs-Integration.
## Ziel
Erstelle umfassende, umsetzbare Erkenntnisse durch die Synthese mehrerer Datenströme.
## Prinzipien
- Integrieren: Verknüpfe Erkenntnisse aus Metriken, Aktivitäten und Physiologie.
- Kontextualisieren: Beziehe Daten auf die Geschichte und Ziele des Athleten.
- Vereinfachen: Mache komplexe Zusammenhänge verständlich."""

SYNTHESIS_PLOT_INSTRUCTIONS = """
## Diagramm-Integration
- Der Abschnitt **„Diagramme in dieser Ausführung“** im Nutzer-Prompt ist maßgeblich: verwende **nur** die dort aufgeführten `[PLOT:…]`-Strings.
- Ignoriere abweichende oder erfundene `[PLOT:…]`-Vorkommen in den Experten-Texten.
- Diese werden zu interaktiven Diagrammen."""

SYNTHESIS_USER_PROMPT_BASE = """{plot_catalog}
Synthetisiere die Expertenanalysen zu einem umfassenden Athletenbericht.

## Inputs
### Athlet
- Name: {athlete_name}
### Metriken
```markdown
{metrics_result}
```
### Aktivität
```markdown
{activity_result}
```
### Physiologie
```markdown
{physiology_result}
```
### Kontext
- Wettkämpfe: ```json {competitions} ```
- Datum: ```json {current_date} ```
- Stil-Leitfaden: ```markdown {style_guide} ```

## Aufgabe
1. **Integrieren**: Verknüpfe Belastung (Metriken), Ausführung (Aktivität) und Reaktion (Physiologie).
2. **Muster identifizieren**: Erkenne Trends in Leistung und Anpassung.
3. **Synthetisieren**: Erstelle eine kohärente Geschichte, nicht nur eine Liste von Fakten.

## Ausgabeformat
- **Header**: Nutze den Athletennamen exakt wie angegeben (keine erfundenen Namen).
- **Executive Summary**: Status auf hoher Ebene und wichtigste Erkenntnisse.
- **Key Performance Indicators**: Tabellenformat.
- **Deep Dive**: Strukturierte Abschnitte mit klaren Überschriften.
- **Empfehlungen**: Kurz und umsetzbar.
- **Tonfall**: Professionell, evidenzbasiert, ermutigend."""

SYNTHESIS_USER_PLOT_INSTRUCTIONS = """
## Diagramm-Referenzen
- Füge jede eindeutige `[PLOT:…]` GENAU EINMAL ein; verwende nur IDs, die bereits in den Experten-Abschnitten vorkommen.
- Erstelle keine Duplikate von Referenzen."""


async def synthesis_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting synthesis node")

    try:
        plotting_enabled = state.get("plotting_enabled", False)
        plot_catalog = (
            format_valid_plot_catalog(state.get("plot_storage_data", {}))
            if plotting_enabled
            else ""
        )

        logger.info(
            "Synthesis node: Plotting %s - %s plot integration instructions",
            "enabled" if plotting_enabled else "disabled",
            "including" if plotting_enabled else "no",
        )

        agent_start_time = datetime.now()

        async def call_synthesis_analysis():
            return await handle_tool_calling_in_node(
                llm_with_tools=ModelSelector.get_llm(AgentRole.SYNTHESIS).bind_tools([]),
                messages=[
                    {"role": "system", "content": (
                        SYNTHESIS_SYSTEM_PROMPT_BASE + (SYNTHESIS_PLOT_INSTRUCTIONS if plotting_enabled else "")
                    )},
                    {"role": "user", "content": (
                        SYNTHESIS_USER_PROMPT_BASE.format(
                            plot_catalog=plot_catalog,
                            athlete_name=state["athlete_name"],
                            metrics_result=extract_expert_output(state.get("metrics_outputs"), "for_synthesis"),
                            activity_result=extract_expert_output(state.get("activity_outputs"), "for_synthesis"),
                            physiology_result=extract_expert_output(state.get("physiology_outputs"), "for_synthesis"),
                            competitions=json.dumps(state["competitions"], indent=2),
                            current_date=json.dumps(state["current_date"], indent=2),
                            style_guide=state["style_guide"],
                        ) + (SYNTHESIS_USER_PLOT_INSTRUCTIONS if plotting_enabled else "")
                    )},
                ],
                tools=[],
                max_iterations=3,
            )

        synthesis_result_raw = await retry_with_backoff(
            call_synthesis_analysis, AI_ANALYSIS_CONFIG, "Synthesis Analysis with Tools"
        )

        # handle_tool_calling_in_node returns (response, usage)
        synthesis_result, usage = synthesis_result_raw

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("Synthesis analysis completed in %.2fs", execution_time)

        return {
            "synthesis_result": synthesis_result,
            "synthesis_complete": True,
            "costs": [
                {
                    "agent": "synthesis",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "usage_metadata": extract_usage_metadata(synthesis_result, AgentRole.SYNTHESIS, usage),
        }

    except Exception as exc:
        logger.exception("Synthesis node failed")
        return {"errors": [f"Synthesis analysis failed: {exc!s}"]}
