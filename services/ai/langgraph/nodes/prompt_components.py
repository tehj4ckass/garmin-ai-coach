from typing import Literal

AgentType = Literal[
    "metrics_summarizer",
    "physiology_summarizer",
    "activity_summarizer",
    "metrics",
    "physiology",
    "activity",
    "synthesis",
    "season_planner",
    "weekly_planner",
]


def get_workflow_context(agent_type: AgentType) -> str:
    # Summarizer agents
    if agent_type in ["metrics_summarizer", "physiology_summarizer", "activity_summarizer"]:
        domain = agent_type.replace("_summarizer", "")
        return f"""
## System-Rolle
Du bist der **{agent_type.replace('_', ' ').title()}**.
- **Input**: Rohdaten `garmin_data`
- **Output**: Strukturierte `{domain}_summary`
- **Ziel**: Fasse Rohdaten in eine sachliche, strukturierte Zusammenfassung für den {domain}-Experten zusammen. NICHT interpretieren."""

    # Expert agents
    if agent_type in ["metrics", "physiology", "activity"]:
        return f"""
## System-Rolle
Du bist der **{agent_type.title()}-Experte**.
- **Input**: `{agent_type}_summary`
- **Output**: `{agent_type}_outputs` mit 3 Feldern:
  1. `for_synthesis`: Für den umfassenden Bericht.
  2. `for_season_planner`: Strategische Einblicke (12-24 Wochen).
  3. `for_weekly_planner`: Taktische Details (nächste 28 Tage).
- **Ziel**: Analysiere Muster und liefere spezifische Erkenntnisse für jeden Empfänger.
- **Kontext**: Du bist einer von 3 parallelen Experten. Konzentriere dich NUR auf deinen Bereich."""

    # Synthesis agent
    if agent_type == "synthesis":
        return """
## System-Rolle
Du bist der **Synthese-Agent**.
- **Input**: `for_synthesis` Felder von Metrics-, Physiology- und Activity-Experten.
- **Output**: `synthesis_result` (Umfassender Athletenbericht).
- **Ziel**: Integriere Domain-Erkenntnisse in eine kohärente Geschichte. Konzentriere dich auf historische Muster, nicht auf die Zukunftsplanung."""

    # Planner agents
    if agent_type in ["season_planner", "weekly_planner"]:
        timeframe = "12-24 Wochen Strategie" if agent_type == "season_planner" else "Workouts der nächsten 28 Tage"
        return f"""
## System-Rolle
Du bist der **{agent_type.replace('_', ' ').title()}**.
- **Input**: `for_{agent_type}` Felder von Metrics-, Physiology- und Activity-Experten.
- **Output**: `{agent_type.replace('_planner', '_plan')}` ({timeframe}).
- **Ziel**: Übersetze Experten-Erkenntnisse in einen konkreten {timeframe}.
- **Kontext**: Nutze die Expertensignale als deine primären Einschränkungen und Leitfäden."""

    return ""


def get_plotting_instructions(agent_name: str) -> str:
    return f"""
## Visualisierungsregeln
- **Einschränkung**: Erstelle Diagramme NUR für einzigartige Einblicke, die in Standard-Garmin-Berichten nicht sichtbar sind. Max. 2 Diagramme.
- **Referenz**: Du MUSST jedes Diagramm GENAU EINMAL in deinem Text referenzieren mit `[PLOT:{agent_name}_TIMESTAMP_ID]`.
- **Platzierung**: Platziere die Referenz dort, wo sie deine Analyse am besten unterstützt. Nicht wiederholen."""


def get_hitl_instructions(agent_name: str) -> str:
    return """
## Menschliche Interaktion
- **Fragen (HITL)**: Wenn du Klärungsbedarf hast, setze `questions` auf eine Liste von `Question`-Items und lasse das normale Ergebnisfeld leer.
- **Ansonsten**: Liefere KEINE `questions`, sondern das normale Ergebnisfeld deines Nodes:
  - Expert-Nodes: `outputs` (mit `for_synthesis`, `for_season_planner`, `for_weekly_planner`)
  - Planner/Summarizer-Nodes: `content` (String)
- **Kriterien**: Frage nur, wenn die Daten zweideutig sind oder Benutzerpräferenzen erforderlich sind. Frage nicht nach offensichtlichen Informationen.
- **Prozess**: Wenn du Fragen stellst, pausiert deine Ausführung, bis der Benutzer antwortet."""
