# Chainlit Q&A: chat with artifacts from a completed analysis run (see pixi task `qa-chat`).
# Uses ModelSelector + synthesis model; AI_MODE from coach_config.yaml via COACH_CONFIG.

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, NamedTuple

import chainlit as cl
import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from core.config import reload_config
from services.ai.ai_settings import AgentRole, ai_settings
from services.ai.model_config import ModelSelector

logger = logging.getLogger(__name__)

# Repo root = parent of cli/
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COACH_CONFIG = REPO_ROOT / "coach_config.yaml"

# Q&A safeguards (env, optional). Limits cost / runaway generations.
def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default


# Max wall-clock time for one assistant reply (streaming).
QA_STREAM_TIMEOUT_SEC = _env_float("QA_STREAM_TIMEOUT_SEC", 60.0)
# Reject comically large single user messages (protects context + cost).
QA_MAX_USER_MESSAGE_CHARS = _env_int("QA_MAX_USER_MESSAGE_CHARS", 12000)


def _qa_max_output_tokens() -> int:
    """Cap completion length (saves output $); use QA_RESPONSE_BUDGET or QA_MAX_OUTPUT_TOKENS."""
    raw = (os.environ.get("QA_MAX_OUTPUT_TOKENS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("Invalid QA_MAX_OUTPUT_TOKENS=%r; using QA_RESPONSE_BUDGET preset", raw)
    budget = (os.environ.get("QA_RESPONSE_BUDGET") or "medium").strip().lower()
    if budget in ("short", "concise", "klein", "mini"):
        return 896
    if budget in ("long", "ausführlich"):
        return 3072
    if budget in ("full", "max"):
        return 4096
    return 1536


# Resolved once at import (env fixed for Chainlit process).
QA_MAX_OUTPUT_TOKENS = _qa_max_output_tokens()


class QAContextLimits(NamedTuple):
    """Per-artifact char caps for Q&A context (approx. input-token / cost control).

    NamedTuple (not dataclass): Chainlit loads this file via ``exec_module``; under
    Python 3.13, ``@dataclass`` can crash because ``sys.modules`` is not set yet.
    """

    summary_chars: int
    expert_chars: int
    garmin_chars: int
    markdown_chars: int
    html_chars: int


def _qa_context_limits() -> QAContextLimits:
    """Defaults target lower prompt size; use QA_CONTEXT_PROFILE=full for legacy (large) context."""
    raw = (os.environ.get("QA_CONTEXT_PROFILE") or "balanced").strip().lower()
    if raw in ("full", "legacy", "max"):
        base = QAContextLimits(8000, 45000, 100000, 25000, 35000)
    elif raw in ("economy", "cheap", "minimal"):
        base = QAContextLimits(4000, 8000, 12000, 4000, 6000)
    elif raw in ("balanced", "default", ""):
        base = QAContextLimits(6000, 14000, 24000, 8000, 10000)
    else:
        logger.warning("Unknown QA_CONTEXT_PROFILE=%r; using balanced", raw)
        base = QAContextLimits(6000, 14000, 24000, 8000, 10000)
    return QAContextLimits(
        summary_chars=_env_int("QA_CONTEXT_SUMMARY_MAX_CHARS", base.summary_chars),
        expert_chars=_env_int("QA_CONTEXT_EXPERT_MAX_CHARS", base.expert_chars),
        garmin_chars=_env_int("QA_CONTEXT_GARMIN_MAX_CHARS", base.garmin_chars),
        markdown_chars=_env_int("QA_CONTEXT_MARKDOWN_MAX_CHARS", base.markdown_chars),
        html_chars=_env_int("QA_CONTEXT_HTML_MAX_CHARS", base.html_chars),
    )


def _qa_context_profile_label() -> str:
    p = (os.environ.get("QA_CONTEXT_PROFILE") or "balanced").strip() or "balanced"
    return p.lower()


class QAUserCancelled(Exception):
    """User clicked Stop in the Chainlit UI while streaming."""


SYSTEM_PROMPT = """Du bist der **Garmin AI Coach - Q&A-Assistent**.

## Kontext
Der Nutzer hat bereits eine vollständige Analyse ausgeführt. Dir liegen strukturierte Ergebnisse (Experten-JSON, optional Rohdaten) vor.

## Länge (verbindlich)
- Antworte **knapp**: Lieber **3–8 knackige Bulletpoints** oder **2 kurze Absätze** als lange Essays.
- **Keine** Wiederholungen, keine „Einleitung/Fazit“-Floskeln, kein erfundener Mehrwert-Text.
- **Details nur**, wenn die Frage sie braucht — oder mit einem Satz anbieten: *„Soll ich X vertiefen?“*
- Zahlen/Belege **gezielt** (1–2 pro Aussage), nicht die ganze Analyse neu erzählen.

## Regeln
1. Beantworte Fragen **primär auf Basis der mitgelieferten Artefakte**. Zitiere oder paraphrasiere konkrete Punkte (Signals, Evidence, Zahlen).
2. Wenn etwas **nicht** in den Daten steht, sage das klar: „Dazu liegen in diesem Run keine Daten vor.“
3. Du darfst **allgemeines Trainingswissen** nutzen, um Vorschläge zu formulieren — kennzeichne das als **„Coaching-Hypothese“** oder **„allgemeine Empfehlung“**, wenn es nicht direkt aus den Artefakten folgt.
4. Antworte auf **Deutsch**, sachlich und ermutigend.
5. Bei konkreten Workout-Vorschlägen: Dauer, Intensität (Zone/RPE/HR wo sinnvoll), Pausen — kurz; erwähne **Risiken**, wenn der Plan von der ursprünglichen Empfehlung abweicht.

## Format
- Zuerst die **Kernantwort** (1–2 Sätze oder 3 bullets), nur bei Bedarf **„Details:“** mit Unterpunkten.
- Wenn sinnvoll: **Belege** mit Verweis auf die Quelle (z. B. "metrics_expert.json - for_weekly_planner").
"""


def _technical_meta_block() -> str:
    """Injected so the model can answer 'which model / mode' from prompt facts."""
    return f"""
## Technische Meta (bei Meta-Fragen verbindlich nutzen)
- **AI_MODE:** `{get_ai_mode_label()}`
- **Konfiguriertes Synthese-Modell:** `{get_synthesis_model_id()}`
- **Q&A-Absicherung:** Stream-Timeout ca. {QA_STREAM_TIMEOUT_SEC:.0f}s; Output-Tokens typisch capped auf ca. {QA_MAX_OUTPUT_TOKENS} (siehe Server).
"""


def _project_data_root() -> Path:
    raw = os.environ.get("GARMIN_COACH_DATA") or os.environ.get("COACH_DATA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    cfg = os.environ.get("COACH_CONFIG", str(DEFAULT_COACH_CONFIG))
    p = Path(cfg).expanduser().resolve()
    if p.exists():
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            out = (doc.get("output") or {}).get("directory", "./data")
            return (p.parent / str(out).lstrip("./")).resolve()
        except Exception:
            logger.debug("Could not read output.directory from %s", p)
    return (REPO_ROOT / "data").resolve()


def apply_coach_config_ai_mode() -> str:
    """Set AI_MODE from coach_config.yaml (extraction.ai_mode) and reload settings."""
    cfg_path = Path(os.environ.get("COACH_CONFIG", str(DEFAULT_COACH_CONFIG))).expanduser().resolve()
    if cfg_path.exists():
        doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        mode = (doc.get("extraction") or {}).get("ai_mode")
        if mode is not None:
            m = str(mode).strip().lower()
            if m == "pro":
                m = "gemini_pro"
            os.environ["AI_MODE"] = m
    qa_override = (os.environ.get("QA_AI_MODE") or "").strip()
    if qa_override:
        m = qa_override.lower()
        if m == "pro":
            m = "gemini_pro"
        os.environ["AI_MODE"] = m
    reload_config()
    ai_settings.reload()
    return get_ai_mode_label()


def get_ai_mode_label() -> str:
    from core.config import get_config

    return get_config().ai_mode.value


def get_synthesis_model_id() -> str:
    return ai_settings.get_model_for_role(AgentRole.SYNTHESIS)


def _build_qa_llm():
    """Synthesis model with optional output token cap for Q&A cost control."""
    base = ModelSelector.get_llm(AgentRole.SYNTHESIS)
    if QA_MAX_OUTPUT_TOKENS <= 0:
        return base
    try:
        return base.bind(max_tokens=QA_MAX_OUTPUT_TOKENS)
    except Exception as e:
        logger.warning("Q&A: max_tokens bind failed (%s); using base LLM", e)
        return base


def _safe_read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read JSON %s: %s", path, e)
        return None


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n\n… [Text gekürzt: {len(text)} Zeichen gesamt]", True


def _json_block(label: str, data: Any, max_chars: int) -> str:
    if data is None:
        return f"### {label}\n_(Datei fehlt)_\n\n"
    try:
        raw = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        raw = str(data)
    body, cut = _truncate_text(raw, max_chars)
    note = f"\n_(Gekürzt: ja, max {max_chars} Zeichen)_\n" if cut else ""
    return f"### {label}\n```json\n{body}\n```{note}\n\n"


def _read_text_file(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    try:
        t = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"_(Lesefehler: {e})_"
    body, cut = _truncate_text(t, max_chars)
    note = f"\n\n_(HTML gekürzt: {cut})_" if cut else ""
    return body + note


def build_context_pack(run_dir: Path) -> str:
    """Assemble prompt context from a run folder."""
    lim = _qa_context_limits()
    parts: list[str] = []
    parts.append(f"## Analyse-Run\n**Ordner:** `{run_dir}`\n\n")

    summary = _safe_read_json(run_dir / "summary.json")
    if summary:
        parts.append(_json_block("summary.json", summary, max_chars=lim.summary_chars))

    for name in ("metrics_expert.json", "activity_expert.json", "physiology_expert.json"):
        data = _safe_read_json(run_dir / name)
        if data:
            parts.append(_json_block(name, data, max_chars=lim.expert_chars))

    garmin = _safe_read_json(run_dir / "garmin_data.json")
    if garmin is not None:
        parts.append(
            _json_block(
                "garmin_data.json (extrahierte Roh-/Strukturdaten)",
                garmin,
                max_chars=lim.garmin_chars,
            )
        )
    else:
        parts.append("### garmin_data.json\n_(Nicht vorhanden — älterer Run ohne Persistenz.)_\n\n")

    for md_name, title in (
        ("season_plan.md", "season_plan.md"),
        ("weekly_plan.md", "weekly_plan.md"),
    ):
        p = run_dir / md_name
        if p.exists():
            txt = _read_text_file(p, max_chars=lim.markdown_chars)
            parts.append(f"### {title}\n```markdown\n{txt}\n```\n\n")

    analysis_html = run_dir / "analysis.html"
    if analysis_html.exists():
        hx = _read_text_file(analysis_html, max_chars=lim.html_chars)
        parts.append(f"### analysis.html (Auszug)\n```html\n{hx}\n```\n\n")

    packed = "".join(parts)
    logger.info(
        "Q&A context pack: profile=%s approx_chars=%d (caps: expert=%d garmin=%d html=%d)",
        _qa_context_profile_label(),
        len(packed),
        lim.expert_chars,
        lim.garmin_chars,
        lim.html_chars,
    )
    return packed


def list_recent_runs(data_root: Path, limit: int = 15) -> list[Path]:
    if not data_root.is_dir():
        return []
    runs = [p for p in data_root.iterdir() if p.is_dir()]
    try:
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        runs.sort(key=lambda p: p.name, reverse=True)
    return runs[:limit]


def parse_run_choice(text: str, indexed_runs: list[Path], data_root: Path) -> Path | None:
    t = text.strip().strip('"').strip("'")
    # Number from welcome list
    if re.fullmatch(r"\d{1,2}", t):
        idx = int(t) - 1
        if 0 <= idx < len(indexed_runs):
            return indexed_runs[idx]
    # Absolute or relative path
    p = Path(t).expanduser()
    if not p.is_absolute():
        p = (data_root / t).resolve()
    else:
        p = p.resolve()
    if not p.is_dir():
        return None
    experts = ("metrics_expert.json", "activity_expert.json", "physiology_expert.json")
    if (p / "garmin_data.json").exists() or any((p / f).exists() for f in experts):
        return p
    return None


def validate_run_dir(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "Pfad ist kein Ordner."
    markers = [
        path / "garmin_data.json",
        path / "metrics_expert.json",
        path / "activity_expert.json",
        path / "physiology_expert.json",
    ]
    if not any(m.exists() for m in markers):
        return False, "Im Ordner fehlen erwartete Dateien (z. B. garmin_data.json oder *_expert.json)."
    return True, ""


@cl.on_chat_start
async def on_chat_start() -> None:
    apply_coach_config_ai_mode()
    model = get_synthesis_model_id()
    mode = get_ai_mode_label()
    data_root = _project_data_root()
    runs = list_recent_runs(data_root)
    cl.user_session.set("indexed_runs", runs)
    cl.user_session.set("data_root", data_root)
    cl.user_session.set("run_dir", None)
    cl.user_session.set("context_pack", None)
    cl.user_session.set("chat_history", [])

    lines = [
        "### Garmin AI Coach · Q&A",
        "",
        f"**AI_MODE:** `{mode}` · **Modell (Synthese):** `{model}`",
    ]
    if (os.environ.get("QA_AI_MODE") or "").strip():
        lines.append(
            "_(**QA_AI_MODE** ist gesetzt: günstigeres/explizites Chat-Tier unabhängig von `coach_config`.)_"
        )
    lines.append(
        f"_(Eingabe-Tokens/Kosten: Kontextprofil **`{_qa_context_profile_label()}`** — `QA_CONTEXT_PROFILE` "
        "`balanced` (Standard) · `economy` · `full`.)_"
    )
    lines.append(
        f"_(Antwortlänge: **~{QA_MAX_OUTPUT_TOKENS}** Output-Tokens cap — `QA_RESPONSE_BUDGET` short · medium · long · full "
        "oder `QA_MAX_OUTPUT_TOKENS`.)_"
    )
    lines.extend(
        [
            "",
            f"Daten-Root: `{data_root}`",
            "",
            "Wähle einen **abgeschlossenen Analyse-Run**, damit ich die Experten-Ergebnisse und Rohdaten einbeziehen kann.",
        ]
    )
    if runs:
        lines.append("")
        lines.append("| # | Run-Ordner |")
        lines.append("|---|------------|")
        for i, r in enumerate(runs, start=1):
            lines.append(f"| **{i}** | `{r.name}` |")
        lines.append("")
        lines.append(
            "Tippe **die Nummer** (z. B. `1`) oder den **Ordnernamen** / den **vollständigen Pfad**."
        )
    else:
        lines.append("")
        lines.append(
            f"Unter `{data_root}` wurden keine Run-Ordner gefunden. "
            "Gib einen **vollständigen Pfad** zu einem Run-Ordner ein (mit `garmin_data.json` oder Expert-JSONs)."
        )
    lines.append("")
    lines.append(
        f"Während eine Antwort **streamt**: **Stop** in der Chainlit-Oberfläche bricht die Generierung ab "
        f"(Timeout: **{QA_STREAM_TIMEOUT_SEC:.0f}s** pro Antwort; optional `QA_MAX_OUTPUT_TOKENS`)."
    )

    await cl.Message(content="\n".join(lines)).send()


def _history_to_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for user_t, assistant_t in history:
        out.append(HumanMessage(content=user_t))
        out.append(AIMessage(content=assistant_t))
    return out


def _chunk_text_from_stream(chunk: Any) -> str:
    """Extract text delta from LangChain stream chunks (provider-agnostic)."""
    if chunk is None:
        return ""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


async def _stream_qa_answer(
    llm: Any,
    messages: list[BaseMessage],
    reply: cl.Message,
    cancel_ev: asyncio.Event,
) -> tuple[str, str | None]:
    """Stream model output into ``reply``.

    Returns (full_text, error_or_none). ``error_or_none`` is set only on hard LLM failures
    (not user stop / timeout).
    """
    full_reply = ""

    async def consume_stream() -> None:
        nonlocal full_reply
        async for chunk in llm.astream(messages):
            if cancel_ev.is_set():
                raise QAUserCancelled()
            piece = _chunk_text_from_stream(chunk)
            if piece:
                full_reply += piece
                await reply.stream_token(piece)

    try:
        await asyncio.wait_for(consume_stream(), timeout=QA_STREAM_TIMEOUT_SEC)
    except QAUserCancelled:
        suffix = (
            "\n\n---\n_(Generierung **gestoppt** (Chainlit Stop). "
            "Bereits erzeugter Text bleibt oben.)_"
        )
        full_reply += suffix
        await reply.stream_token(suffix)
    except TimeoutError:
        logger.warning("Q&A stream hit QA_STREAM_TIMEOUT_SEC=%s", QA_STREAM_TIMEOUT_SEC)
        suffix = (
            f"\n\n---\n_(**Timeout** nach {QA_STREAM_TIMEOUT_SEC:.0f}s — "
            "Generierung abgebrochen, um Kosten zu begrenzen.)_"
        )
        full_reply += suffix
        await reply.stream_token(suffix)
    except Exception as e:
        logger.exception("LLM stream failed")
        return full_reply, str(e)
    return full_reply, None


@cl.on_stop
async def on_stop() -> None:
    """Chainlit Stop button: request cancellation of the current streamed reply."""
    ev = cl.user_session.get("qa_cancel_stream")
    if ev is not None:
        ev.set()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = (message.content or "").strip()
    if not text:
        return

    data_root: Path = cl.user_session.get("data_root") or _project_data_root()
    indexed: list[Path] = cl.user_session.get("indexed_runs") or []

    run_dir: Path | None = cl.user_session.get("run_dir")
    if run_dir is None:
        choice = parse_run_choice(text, indexed, data_root)
        if choice is None:
            await cl.Message(
                content=(
                    "Das habe ich nicht als gültigen Run erkannt. "
                    "Bitte eine **Nummer** aus der Liste, den **Ordnernamen** unter `data/`, oder einen **vollständigen Pfad**."
                )
            ).send()
            return
        ok, err = validate_run_dir(choice)
        if not ok:
            await cl.Message(content=f"❌ {err}").send()
            return
        cl.user_session.set("run_dir", choice)
        context = build_context_pack(choice)
        cl.user_session.set("context_pack", context)
        await cl.Message(
            content=(
                f"✅ Kontext geladen: `{choice.name}`\n\n"
                f"**AI_MODE:** `{get_ai_mode_label()}` · **Modell:** `{get_synthesis_model_id()}`\n\n"
                "Stelle jetzt deine Fragen zur Analyse, zum Training oder zu den Empfehlungen."
            )
        ).send()
        return

    if len(text) > QA_MAX_USER_MESSAGE_CHARS:
        await cl.Message(
            content=(
                f"Die Nachricht ist zu lang (**{len(text)}** Zeichen; Maximum **{QA_MAX_USER_MESSAGE_CHARS}**). "
                "Kürze die Frage oder teile sie in mehrere Nachrichten."
            )
        ).send()
        return

    context_pack: str = cl.user_session.get("context_pack") or ""
    history: list[tuple[str, str]] = cl.user_session.get("chat_history") or []

    llm = _build_qa_llm()
    sys_content = (
        SYSTEM_PROMPT
        + _technical_meta_block()
        + "\n\n## Artefakte dieses Runs\n\n"
        + context_pack
    )

    messages: list[BaseMessage] = [SystemMessage(content=sys_content)]
    messages.extend(_history_to_messages(history[-6:]))
    messages.append(HumanMessage(content=text))

    reply = cl.Message(content="")
    await reply.send()

    cancel_ev = asyncio.Event()
    cl.user_session.set("qa_cancel_stream", cancel_ev)
    try:
        full_reply, stream_error = await _stream_qa_answer(llm, messages, reply, cancel_ev)
    finally:
        cl.user_session.set("qa_cancel_stream", None)

    if stream_error is not None:
        await cl.Message(content=f"❌ Fehler beim Aufruf des Modells: {stream_error}").send()
        return

    history.append((text, full_reply))
    cl.user_session.set("chat_history", history)


def main() -> None:
    """Allow `python -m cli.qa_chainlit_app` to print hint."""
    print("Start with: chainlit run cli/qa_chainlit_app.py")
    print("Or: pixi run qa-chat")


if __name__ == "__main__":
    main()
