# Garmin AI Coach · Coach Chat (*talk to your training data*)

Chatte **nach einem abgeschlossenen Run** mit deinen Analyse-Artefakten (Experten-JSON, `garmin_data.json`, optional Pläne & Report). Im README: **Coach Chat** — *talk to your training data*.

## Start

```bash
pixi run qa-chat
```

`AI_MODE` kommt aus `coach_config.yaml` (`extraction.ai_mode`), wie beim Haupt-CLI — oder setze `COACH_CONFIG` / `AI_MODE` in der Umgebung.

**Kosten / Laufzeit:** Während die Antwort streamt, **Stop** in der UI nutzen. Serverseitig gibt es ein **Timeout** pro Antwort (`QA_STREAM_TIMEOUT_SEC`, Standard 180s) und ein **Output-Token-Limit** (`QA_MAX_OUTPUT_TOKENS`, Standard 4096). Siehe `cli/README.md`.

## Run wählen

Beim ersten Schritt: **Nummer** aus der Tabelle, **Ordnername** unter deinem `data/`-Root, oder **vollständiger Pfad** zum Run-Ordner.
