import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import extract_usage_metadata
from .prompt_components import format_valid_plot_catalog, get_report_css
from .tool_calling_helper import extract_text_content

logger = logging.getLogger(__name__)

FORMATTER_SYSTEM_PROMPT = """Du bist ein Design-Technologe für sportliche Leistungsberichte.
## Ziel
Erstelle ein vollständiges HTML-Dokument unter Verwendung des bereitgestellten CSS-Design-Systems.
## Prinzipien
- Verwende AUSSCHLIESSLICH die CSS-Klassen aus dem bereitgestellten Stylesheet.
- Erfinde KEIN eigenes CSS. Das Stylesheet wird als <style>-Block inline eingefügt.
- Klarheit: Sofortiges Verständnis durch visuelle Hierarchie.
- Vollständigkeit: ALLE Inhalte, Metriken und Scores müssen enthalten sein."""

FORMATTER_USER_PROMPT_BASE = """Transformiere den folgenden Analyse-Inhalt in ein HTML-Dokument.

{plot_catalog}

## Design-System CSS (PFLICHT — als <style>-Block in <head> einfügen)
```css
{report_css}
```

## Inhalt (Markdown → HTML umwandeln)
```markdown
{synthesis_result}
```

## Struktur-Anweisungen
Verwende diese Klassen aus dem CSS:
- `.report-container` — äußerer Wrapper (max-width 1100px)
- `.report-header` mit `.subtitle` ("PERFORMANCE ANALYSIS"), `h1` (Athletenname: "{athlete_name}"), `.meta-row` mit `.meta-item` (Datum, AI-Mode, Level)
- `.section` mit `.section-label` (nummeriert: `<span class="section-num">01</span> Overview`) und `.section-title`
- `.card` mit `.card-header` → `.card-title` + `.tag` (tag-accent/tag-cyan/tag-amber/tag-rose/tag-blue)
- `.grid .grid-2` / `.grid-3` / `.grid-4` für Spalten-Layouts
- `.kpi` innerhalb von `.card` → `.kpi-value`, `.kpi-label`, `.kpi-delta.positive/.negative/.neutral`
- `.data-table` mit `<th>` (Monospace-Header) und `<td>`
- `.callout` / `.callout-warn` / `.callout-danger` / `.callout-info` für Alerts
- `<ul>` / `<ol>` für Listen (Marker-Farbe ist automatisch accent)
- `.report-footer` am Ende
- Emojis in `.card-title` und `.section-label` als visuelle Marker

## Regeln
1. Füge das CSS EXAKT so als `<style>`-Block im `<head>` ein. KEIN zusätzliches CSS.
2. Die `@import`-Zeile für Google Fonts MUSS als erstes im `<style>`-Block stehen.
3. Alle Metriken und Scores aus dem Markdown müssen enthalten sein.
4. Gib NUR das vollständige HTML-Dokument zurück."""

FORMATTER_PLOT_INSTRUCTIONS = """
## Diagramm-Integration
- **Beibehalten**: Behalte `[PLOT:…]`-Referenzen EXAKT bei — **nur** die IDs aus dem Katalog oben (falls vorhanden). Keine neuen Namen erfinden.
- **Layout**: Behandle sie als große visuelle Blöcke (volle Breite) innerhalb einer `.card`.
- **Abstände**: Stelle sicher, dass die Card vertikalen Platz (~500px) für die interaktiven Diagramme vorsieht, die sie ersetzen werden."""


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

        plot_catalog = (
            format_valid_plot_catalog(state.get("plot_storage_data", {}))
            if plotting_enabled
            else ""
        )

        async def call_html_formatting():
            synthesis_result = extract_text_content(state.get("synthesis_result", ""))

            response = await ModelSelector.get_llm(AgentRole.FORMATTER).ainvoke([
                {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    FORMATTER_USER_PROMPT_BASE.format(
                        plot_catalog=plot_catalog,
                        report_css=get_report_css(),
                        synthesis_result=synthesis_result,
                        athlete_name=state.get("athlete_name", "Athlete"),
                    )
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
