# Garmin AI Coach · Coach Chat (*talk to your training data*)

Chatte **nach einem abgeschlossenen Run** mit deinen Analyse-Artefakten (Experten-JSON, `garmin_data.json`, optional Pläne & Report). Im README: **Coach Chat** — *talk to your training data*.

## Start

```bash
pixi run qa-chat
```

`AI_MODE` kommt aus `coach_config.yaml` (`extraction.ai_mode`), wie beim Haupt-CLI — oder setze `COACH_CONFIG` / `AI_MODE` in der Umgebung.

**Kosten / Laufzeit:** Die **Prompt-Größe** treibt den Großteil der Kosten; die **Antwortlänge** spielt über Completion-Tokens noch einmal rein — Standard ist **`QA_RESPONSE_BUDGET=medium`** mit knappem System-Prompt; `QA_RESPONSE_BUDGET=short` / explizites `QA_MAX_OUTPUT_TOKENS` siehe `cli/README.md`. Außerdem: **`QA_CONTEXT_PROFILE`**, **`QA_AI_MODE`**. **Stop** in der UI; `QA_STREAM_TIMEOUT_SEC`.

## Run wählen

Beim ersten Schritt: **Nummer** aus der Tabelle, **Ordnername** unter deinem `data/`-Root, oder **vollständiger Pfad** zum Run-Ordner.
