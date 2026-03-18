from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState

from .data_summarizer_node import create_data_summarizer_node

ACTIVITY_SUMMARIZER_SYSTEM_PROMPT = """## Ziel
Extrahiere und strukturiere Trainingsaktivitätsdaten mit sachlicher Präzision.
## Prinzipien
- Sei objektiv: Präsentiere Daten ohne Interpretation.
- Sei präzise: Behalte exakte Metriken und Einheiten bei.
- Sei strukturiert: Nutze konsistente Formatierung und transparente Komprimierung."""

ACTIVITY_SUMMARIZER_USER_PROMPT = """## Aufgabe
Beschreibe objektiv die jüngsten Trainingsaktivitäten des Athleten.

## Einschränkungen
- STRENGSTENS KEINE Interpretation oder Coaching-Ratschläge.
- Nutze transparente Komprimierung für lange Zeitreihen (fasse repetitive Muster in Fenstern oder Bereichen zusammen).

## Erforderliche Struktur
1. **Tabelle aller Aktivitäten**: kompakte Tabelle für jede Aktivität (Datum, Typ, Dauer, Distanz, Höhenmeter, durchschn. HF, durchschn. Pace/Leistung).
2. **Schlüsselsitzungen**: Deep Dives NUR für Schlüsselsitzungen (Intensität/Neuheit/Anomalie).
3. **Zonenverteilungen**: Fasse Verteilungen in Tabellen zusammen.

## Input-Daten
```json
{data}
```

## Vorlage für Schlüsselsitzungen
# Aktivität: [Datum - Typ]

## Übersicht
* Dauer: [Zeit]
* Distanz: [Distanz]
* Höhenmeter: [Höhenmeter]
* Durchschn. HF: [HF] | Durchschn. Pace/Leistung: [Pace/Leistung]

## Runden-Details
| Runde | Dist | Zeit | Pace | Durchschn. HF | Max. HF | ... |
|-------|------|------|------|---------------|---------|-----|
| 1     | ...  | ...  | ...  | ...           | ...     | ... |"""


def extract_activity_data(state: TrainingAnalysisState) -> dict:
    return state["garmin_data"].get("recent_activities", [])


activity_summarizer_node = create_data_summarizer_node(
    node_name="Activity Summarizer",
    agent_role=AgentRole.SUMMARIZER,
    data_extractor=extract_activity_data,
    state_output_key="activity_summary",
    agent_type="activity_summarizer",
    system_prompt=ACTIVITY_SUMMARIZER_SYSTEM_PROMPT,
    user_prompt=ACTIVITY_SUMMARIZER_USER_PROMPT,
)
