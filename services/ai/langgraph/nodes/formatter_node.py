import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import extract_usage_metadata
from .tool_calling_helper import extract_text_content

logger = logging.getLogger(__name__)

FORMATTER_SYSTEM_PROMPT = """Du bist ein Design-Technologe.
## Ziel
Erstelle ansprechende, funktionale HTML-Dokumente für sportliche Leistungsdaten.
## Prinzipien
- Klarheit: Design für sofortiges Verständnis.
- Hierarchie: Nutze visuelle Strukturen, um die Aufmerksamkeit zu lenken.
- Ästhetik: Balance zwischen Schönheit und Funktion."""

FORMATTER_USER_PROMPT_BASE = """Transformiere diesen Inhalt in ein schönes HTML-Dokument.

## Inhalt
```markdown
{synthesis_result}
```

## Aufgabe
Erstelle ein vollständiges HTML-Dokument mit:
1. **Struktur**: Logische Organisation mit klaren Überschriften.
2. **Design**: Sauberes CSS, responsives Layout, professionelle Typografie.
3. **Visuelle Elemente**: Nutze Emojis und Farben, um Daten hervorzuheben (z. B. 🎯 Ziele, 📊 Metriken).
4. **Vollständigkeit**: Füge ALLE Inhalte, Metriken und Scores ein.

## Ausgabe
Gib NUR das vollständige HTML-Dokument zurück."""

FORMATTER_PLOT_INSTRUCTIONS = """
## Diagramm-Integration
- **Beibehalten**: Behalte `[PLOT:plot_id]`-Referenzen EXAKT so bei, wie sie geschrieben wurden.
- **Layout**: Behandle sie als große visuelle Blöcke (volle Breite).
- **Abstände**: Stelle sicher, dass das CSS vertikalen Platz (~500px) für die interaktiven Diagramme vorsieht, die sie ersetzen werden."""


async def formatter_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting HTML formatter node")

    try:
        plotting_enabled = state.get("plotting_enabled", False)
        logger.info(
            "Formatter node: Plotting %s - %s plot integration instructions",
            "enabled" if plotting_enabled else "disabled",
            "including" if plotting_enabled else "no",
        )

        agent_start_time = datetime.now()

        async def call_html_formatting():
            synthesis_result = extract_text_content(state.get("synthesis_result", ""))

            response = await ModelSelector.get_llm(AgentRole.FORMATTER).ainvoke([
                {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    FORMATTER_USER_PROMPT_BASE.format(synthesis_result=synthesis_result)
                    + (FORMATTER_PLOT_INSTRUCTIONS if plotting_enabled else "")
                )},
            ])
            return response

        response = await retry_with_backoff(
            call_html_formatting, AI_ANALYSIS_CONFIG, "HTML Formatting"
        )
        analysis_html = extract_text_content(response)

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("HTML formatting completed in %.2fs", execution_time)

        return {
            "analysis_html": analysis_html,
            "costs": [
                {
                    "agent": "formatter",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "usage_metadata": extract_usage_metadata(response, AgentRole.FORMATTER),
        }

    except Exception as exc:
        logger.exception("Formatter node failed")
        return {"errors": [f"HTML formatting failed: {exc!s}"]}
