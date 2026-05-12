import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import extract_usage_metadata
from .prompt_components import get_report_css
from .tool_calling_helper import extract_text_content

logger = logging.getLogger(__name__)

PLAN_FORMATTER_SYSTEM_PROMPT = """Du bist ein Spezialist für Trainingsplan-Visualisierung.
## Ziel
Transformiere Trainingspläne in HTML-Dokumente unter Verwendung des bereitgestellten CSS-Design-Systems.
## Prinzipien
- Verwende AUSSCHLIESSLICH die CSS-Klassen aus dem bereitgestellten Stylesheet.
- Erfinde KEIN eigenes CSS. Das Stylesheet wird als <style>-Block inline eingefügt.
- Klarheit: Komplexe Trainingsinformationen sofort zugänglich machen.
- Benutzbarkeit: Design für Desktop-Planung und mobile Ausführung.

## Interaktive Checklisten
- Füge für jedes Workout eine Checkbox mit Klasse `workout-check` und passendem `name`-Attribut hinzu.
- Schließe jede Checkbox in ein `<label>` ein für tippfreundliche Interaktion."""

PLAN_FORMATTER_USER_PROMPT = """Transformiere den Trainingsplan in ein HTML-Dokument.

## Design-System CSS (PFLICHT — als <style>-Block in <head> einfügen)
```css
{report_css}
```

## Inputs
### Saisonplan
```markdown
{season_plan}
```
### 4-Wochen-Plan
```markdown
{weekly_plan}
```

## Struktur-Anweisungen
Verwende diese Klassen aus dem CSS:
- `.report-container` — äußerer Wrapper
- `.report-header` mit `.subtitle` ("TRAINING PLAN"), `h1` (Athletenname), `.meta-row` (Zeitraum, Nächstes Event, Level)
- `.section` mit `.section-label` + `.section-title`
- `.card` für Blöcke, `.grid .grid-2` für Spalten
- `.data-table` für Periodisierungsübersicht (Woche/Fokus/TSS/Typ)
- `.tag` mit Varianten: `.tag-accent` (build), `.tag-amber` (overload), `.tag-blue` (recover), `.tag-rose` (A-Priority), `.tag-cyan` (joint session)
- `.week-grid` — 7-spaltiges Grid für Tagesblöcke
- `.day-card` für einzelne Tage, `.day-card.key-session` für Key Sessions, `.day-card.today` für heute
- `.day-label` für Tag/Datum
- `<input type="checkbox" class="workout-check" name="wk1-mon">` für interaktive Checkboxen
- `.callout` / `.callout-info` / `.callout-warn` für Coaching-Hinweise
- `.report-footer` am Ende

## Regeln
1. Füge das CSS EXAKT so als `<style>`-Block im `<head>` ein. KEIN zusätzliches CSS.
2. Die `@import`-Zeile für Google Fonts MUSS als erstes im `<style>`-Block stehen.
3. Der Plan muss KOMPAKT sein — der Benutzer muss das Gesamtbild leicht erfassen können.
4. Behalte ALLE Workout-Details bei, formatiere sie aber dicht in den Day-Cards.
5. Gib NUR das vollständige HTML-Dokument zurück.
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
                    report_css=get_report_css(),
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
