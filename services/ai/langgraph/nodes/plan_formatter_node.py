import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import extract_usage_metadata
from .tool_calling_helper import extract_text_content

logger = logging.getLogger(__name__)

PLAN_FORMATTER_SYSTEM_PROMPT = """Du bist ein Spezialist für Datenvisualisierung.
## Ziel
Transformiere Trainingspläne in ansprechende, funktionale HTML-Dokumente.
## Prinzipien
- Klarheit: Mache komplexe Trainingsinformationen sofort zugänglich.
- Hierarchie: Nutze visuelle Strukturen, um die Aufmerksamkeit zu lenken.
- Benutzbarkeit: Design für Desktop-Planung und mobile Ausführung.
- Ästhetik: Erstelle ein professionelles, athletenorientiertes visuelles Erlebnis.

## Interaktive Checklisten
- Füge für jedes Workout und jede Teilaufgabe eine native HTML-Checkbox mit <input type="checkbox"> hinzu, damit der Benutzer Elemente direkt im Browser abhaken kann.
- Schließe jede Checkbox in ein <label> ein (oder verknüpfe sie über for/id) für eine tippfreundliche, zugängliche Interaktion.
- Verwende aussagekräftige name/value-Attribute (z. B. name="wk-2025-09-18-run" value="done"), um optionales Absenden von Formularen zu unterstützen."""

PLAN_FORMATTER_USER_PROMPT = """Transformiere den Trainingsplan in ein professionelles HTML-Dokument.

## Inputs
### Saisonplan
```markdown
{season_plan}
```
### 4-Wochen-Plan
```markdown
{weekly_plan}
```

## Aufgabe
Konvertiere den Markdown-Inhalt in ein einzelnes, eigenständiges HTML-Dokument.

## Einschränkungen
- **Kompaktheit**: Der Benutzer muss das "Gesamtbild" leicht erfassen können. Vermeide übermäßiges Scrollen.
- **Layout**: Nutze ein dichtes, informationsreiches Layout (z. B. Grid oder kompakte Karten) für den 4-Wochen-Plan.
- **Benutzbarkeit**: Füge interaktive Checkboxen für jedes Workout-Element hinzu.
- **Design**: Professionelle, athletenorientierte Ästhetik mit klarer visueller Hierarchie.

## Ausgabeanforderungen
1. **Struktur**:
   - Header: Name des Athleten und Zeitraum.
   - Abschnitt 1: Saisonplan-Übersicht (Übergeordnet).
   - Abschnitt 2: 4-Wochen-Plan (Detailliert, aber kompakt).
2. **Format**: Vollständiges HTML5-Dokument mit eingebettetem CSS.
3. **Inhalt**: Behalte alle Workout-Details bei, aber formatiere sie dicht.
4. **Rückgabe**: NUR den HTML-Code.
"""


async def plan_formatter_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting plan formatter node")

    try:
        agent_start_time = datetime.now()

        def get_content(field):
            value = state.get(field, "")
            if hasattr(value, "content") or hasattr(value, "questions"):
                if getattr(value, "questions", None):
                    raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")
                return getattr(value, "content", "") or ""
            if hasattr(value, "output"):
                # Backwards compatibility
                output = value.output
                if isinstance(output, str):
                    return output
                raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")
            if isinstance(value, dict):
                if value.get("questions"):
                    raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")
                return value.get("content", value.get("output", value))
            return value

        async def call_plan_formatting():
            response = await ModelSelector.get_llm(AgentRole.FORMATTER).ainvoke([
                {"role": "system", "content": PLAN_FORMATTER_SYSTEM_PROMPT},
                {"role": "user", "content": PLAN_FORMATTER_USER_PROMPT.format(
                    season_plan=get_content("season_plan"),
                    weekly_plan=get_content("weekly_plan")
                )},
            ])
            return response

        response = await retry_with_backoff(
            call_plan_formatting, AI_ANALYSIS_CONFIG, "Plan Formatter"
        )
        planning_html = extract_text_content(response)

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("Plan formatting completed in %.2fs", execution_time)

        return {
            "planning_html": planning_html,
            "costs": [
                {
                    "agent": "plan_formatter",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "usage_metadata": extract_usage_metadata(response, AgentRole.FORMATTER),
        }

    except Exception as exc:
        logger.exception("Plan formatter node failed")
        return {"errors": [f"Plan formatting failed: {exc!s}"]}
