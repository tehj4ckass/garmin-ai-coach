import functools
import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Athlete-Level Coaching Overlays
# ---------------------------------------------------------------------------
# Each overlay adjusts tone, detail depth, metric focus, and recommendation
# style for all expert / planner / synthesis prompts.  Injected via
# get_level_context() — one call per node, no prompt duplication.
# ---------------------------------------------------------------------------

LEVEL_OVERLAYS: dict[str, str] = {
    "beginner": """
## Athleten-Level: Beginner / Hobbysportler
- **Tonfall**: Ermutigend, motivierend, einfache Sprache. Vermeide Fachjargon oder erkläre ihn sofort in einem kurzen Nebensatz.
- **Analyse-Fokus**: Konsistenz und Regelmäßigkeit sind wichtiger als Optimierung. Positive Fortschritte hervorheben.
- **Metriken**: Nur die wichtigsten KPIs (Dauer, Häufigkeit, subjektives Empfinden). ACWR/TSB/Monotonie nur erwähnen, wenn wirklich auffällig — dann allgemeinverständlich erklären (z. B. „Du hast letzte Woche deutlich mehr trainiert als üblich").
- **Empfehlungen**: Einfach, konkret, maximal 3 Handlungsempfehlungen. Keine Mikrooptimierung. „Du könntest…" statt „Du musst…".
- **Risiko-Bewertung**: Konservativ, aber nicht angstmachend. Überlastungssignale freundlich ansprechen.
- **Planungsstil**: Flexible Vorschläge statt strikter Vorgaben. Spaß und Gesundheit priorisieren.
""",
    "advanced": """
## Athleten-Level: Advanced Amateur
- **Tonfall**: Sachlich-freundlich, coach-ähnlich. Fachbegriffe verwenden, aber nicht überfrachten.
- **Analyse-Fokus**: Balance zwischen Muster-Erkennung und praktischer Umsetzbarkeit. Periodisierung berücksichtigen.
- **Metriken**: Vollständige KPI-Analyse inklusive ACWR, Belastungstrends, HRV-Muster. Einordnung mit Kontext.
- **Empfehlungen**: Strukturiert und begründet, mit Alternativen. Erklären warum, nicht nur was.
- **Risiko-Bewertung**: Differenziert — zwischen „beobachten" und „handeln" unterscheiden.
- **Planungsstil**: Klare Empfehlungen mit Spielraum für Anpassung. Progression und Erholung ausbalancieren.
""",
    "elite": """
## Athleten-Level: Elite Athlete
- **Tonfall**: Direkt, datengetrieben, ohne Beschönigung. Kritische Analyse erwünscht.
- **Analyse-Fokus**: Maximale Detailtiefe. Subtile Muster, marginale Gewinne, Crash-Signaturen erkennen.
- **Metriken**: Alle verfügbaren KPIs mit voller ACWR-/Monotonie-/Strain-Analyse. Schwellenwerte benennen und bewerten.
- **Empfehlungen**: Präzise, evidenzbasiert, mit Risiko-Reward-Abwägung. Grenzbelastung diskutieren.
- **Risiko-Bewertung**: Schonungslos ehrlich. Overreaching vs. Overtraining differenzieren.
- **Planungsstil**: Detaillierte Vorgaben mit Zonen, Intensitäten, Progression. Kompromisslos am Ziel orientiert.
""",
}


def get_level_context(level: str) -> str:
    """Return the athlete-level coaching overlay for prompt injection.

    Called by every expert / planner / synthesis node to calibrate the
    LLM's tone, detail depth, and recommendation style.
    """
    return LEVEL_OVERLAYS.get(level, LEVEL_OVERLAYS["advanced"])


# ---------------------------------------------------------------------------
# Report Design System (CSS)
# ---------------------------------------------------------------------------
# Loaded once from docs/report_theme.css and cached.  Injected into the
# formatter prompts so every generated report uses the same visual design.
# ---------------------------------------------------------------------------

_THEME_CSS_PATH = Path(__file__).resolve().parents[4] / "docs" / "report_theme.css"


@functools.lru_cache(maxsize=1)
def get_report_css() -> str:
    """Return the shared report CSS stylesheet for HTML formatter prompts.

    The CSS is read once from ``docs/report_theme.css`` and cached for the
    lifetime of the process.  If the file is missing a warning is logged and
    an empty string is returned so that the formatter still works (just
    without a fixed design).
    """
    if _THEME_CSS_PATH.is_file():
        css = _THEME_CSS_PATH.read_text(encoding="utf-8")
        logger.debug("Loaded report theme CSS (%d chars) from %s", len(css), _THEME_CSS_PATH)
        return css
    logger.warning("Report theme CSS not found at %s — formatter will use its own styling", _THEME_CSS_PATH)
    return ""

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


def format_valid_plot_catalog(plot_storage_data: dict[str, dict[str, Any]]) -> str:
    """Listet alle gültigen [PLOT:…]-Strings für Synthese/Formatter (State plot_storage_data)."""
    if not plot_storage_data:
        return """## Diagramme in dieser Ausführung
Es liegen **keine** erzeugten Diagramme vor. Setze **keine** `[PLOT:…]`-Platzhalter."""

    lines = [
        "## Diagramme in dieser Ausführung (nur diese `[PLOT:…]`-Strings sind gültig)",
        "",
    ]
    for plot_id, meta in sorted(
        plot_storage_data.items(),
        key=lambda kv: kv[1].get("created_at") or "",
    ):
        desc = (meta.get("description") or "").strip() or "(ohne Beschreibung)"
        agent = meta.get("agent_name") or "?"
        lines.append(f"- `[PLOT:{plot_id}]` — {desc} (Quelle: {agent})")
    lines.extend(
        [
            "",
            "**Pflicht:** Verwende ausschließlich die IDs in den Backticks oben (Zeichen für Zeichen). **Verboten** sind erfundene Kurznamen oder beschreibende Slugs als ID.",
        ]
    )
    return "\n".join(lines)


def get_plotting_instructions(agent_name: str) -> str:
    return f"""
## Visualisierungsregeln
- **Einschränkung**: Erstelle Diagramme NUR für einzigartige Einblicke, die in Standard-Garmin-Berichten nicht sichtbar sind. Max. 2 Diagramme.
- **Plot-ID (kritisch)**: Nach jedem erfolgreichen `python_plotting_tool`-Aufruf liefert die Antwort ein Feld `plot_id` und eine Zeile der Form `Reference as [PLOT:…]`. Du MUSST in deinem Text **exakt dieselbe Zeichenkette** wie in `plot_id` verwenden — z. B. `[PLOT:{agent_name}_1735123456789_001]`. **Verboten**: eigene beschreibende Namen wie `activity_intensity_distribution` oder `TIMESTAMP_ID` als Platzhalter; solche Referenzen sind im Bericht ungültig.
- **Referenz**: Jedes erzeugte Diagramm GENAU EINMAL mit `[PLOT:<exakte plot_id aus der Tool-Antwort>]` einbinden.
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
