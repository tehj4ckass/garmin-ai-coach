import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from services.ai.ai_settings import AgentRole, ai_settings
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .prompt_components import AgentType, get_workflow_context
from .tool_calling_helper import extract_text_content

logger = logging.getLogger(__name__)

GENERIC_SUMMARIZER_SYSTEM_PROMPT = """## Ziel
Erhalte entscheidungsrelevante Metriken aus Rohdaten mit transparenter Komprimierung.
## Prinzipien
- Erhalten: Behalte aussagekräftige Zahlen (Messungen, Zählungen, Raten) bei, die nachgelagerte Entscheidungen beeinflussen.
- Erkennen: Unterscheide Signale (Messungen) von Rauschen (IDs, Nullwerte).
- Organisieren: Nutze Tabellen und Listen mit klaren Zeitfenstern.
- Transparente Komprimierung: Du DARFST lange Sequenzen komprimieren, wenn du zeigst, wie und wo.
- Keine versteckte Aggregation: Wenn du eine Sequenz zusammenfasst, lege die Werte oder eine explizite Tabelle von Fenstern offen."""

GENERIC_SUMMARIZER_USER_PROMPT = """## Aufgabe
Extrahiere und organisiere entscheidungsrelevante Metriken aus diesen Daten mit transparenter Komprimierung.

## Einschränkungen
- NICHT interpretieren oder spekulieren.
- Schließe wiederholte Nullwerte und strukturelle IDs aus.
- Du DARFST lange Sequenzen komprimieren, aber zeige wie (Fenster, Bereiche oder Tabellen).

## Erforderliche Struktur
1. **Abdeckungs-Header**: Datumsbereich, Abtast-Granularität, fehlende Zeiträume.
2. **Kern-Tabellen**: Zeitindex → Hauptmessungen.
3. **Veränderungspunkte & Extreme**: Höchst-/Tiefstwerte mit Zeitstempeln.
4. **Hinweise zur Datenqualität**: Lücken, verdächtige Nullen, Ausreißer.

## Input-Daten
```json
{data}
```

## Ausgabeformat
- Markdown-Tabellen für numerische Daten.
- Klare Abschnittsüberschriften.
- Konsistente Einheiten.

Liefere eine kompakte, entscheidungsfokussierte Zusammenfassung mit expliziter Komprimierung."""

SUMMARIZER_FINAL_CHECKLIST = """
## Abschluss-Checkliste
- Nur Fakten (keine Interpretation).
- Transparente Komprimierung (keine versteckte Aggregation).
- Entscheidungsrelevante Zahlen priorisiert."""


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def create_data_summarizer_node(
    node_name: str,
    agent_role: AgentRole,
    data_extractor: Callable[[TrainingAnalysisState], dict[str, Any]],
    state_output_key: str,
    agent_type: AgentType,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> Callable:
    workflow_context = get_workflow_context(agent_type)
    base_system_prompt = system_prompt or GENERIC_SUMMARIZER_SYSTEM_PROMPT
    effective_system_prompt = workflow_context + base_system_prompt + SUMMARIZER_FINAL_CHECKLIST
    effective_user_prompt = user_prompt or GENERIC_SUMMARIZER_USER_PROMPT

    async def summarizer_node(state: TrainingAnalysisState) -> dict[str, list | str]:
        logger.info("Starting %s node", node_name)

        try:
            agent_start_time = datetime.now()

            data_to_summarize = data_extractor(state)

            async def call_llm():
                response = await ModelSelector.get_llm(agent_role).ainvoke(
                    [
                        {"role": "system", "content": effective_system_prompt},
                        {
                            "role": "user",
                            "content": effective_user_prompt.format(
                                data=json.dumps(data_to_summarize, indent=2)
                            ),
                        },
                    ]
                )
                return response

            response = await retry_with_backoff(
                call_llm, AI_ANALYSIS_CONFIG, node_name
            )
            summary = extract_text_content(response)

            # Extract usage metadata
            model_name = ai_settings.get_model_for_role(agent_role)
            usage_metadata = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                input_tokens = _safe_int(getattr(usage, "input_tokens", 0))
                output_tokens = _safe_int(getattr(usage, "output_tokens", 0))
                total_tokens = _safe_int(getattr(usage, "total_tokens", input_tokens + output_tokens))
                usage_metadata[model_name] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }

            execution_time = (datetime.now() - agent_start_time).total_seconds()
            logger.info("%s completed in %.2fs", node_name, execution_time)

            return {
                state_output_key: summary,
                "costs": [
                    {
                        "agent": state_output_key.replace("_summary", "_summarizer"),
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "usage_metadata": usage_metadata,
            }

        except Exception as exc:
            logger.error("%s node failed: %s", node_name, exc)
            return {"errors": [f"{node_name} failed: {exc}"]}

    return summarizer_node
