# Garmin AI Coach CLI (primary interface)

Command-line interface for the AI triathlon coach. Uses a YAML or JSON config file to extract your Garmin data, run multi-agent AI analysis and planning, and save HTML reports.

- CLI script: [cli/garmin_ai_coach_cli.py](cli/garmin_ai_coach_cli.py)
- Config template: [cli/coach_config_template.yaml](cli/coach_config_template.yaml)
- Pixi tasks: [pixi.toml](../pixi.toml)

## Quick Start

Using Pixi (recommended):
```bash
# 1) Create a config template
pixi run coach-init my_config.yaml

# 2) Edit the file with your details (athlete.email, context, etc.)

# 3) Run the analysis and planning
pixi run coach-cli --config my_config.yaml
```

Using Python directly:
```bash
python cli/garmin_ai_coach_cli.py --init-config my_config.yaml
python cli/garmin_ai_coach_cli.py --config my_config.yaml [--output-dir ./data]
```

## Command reference

```bash
python cli/garmin_ai_coach_cli.py --config PATH [--output-dir PATH]
python cli/garmin_ai_coach_cli.py --init-config PATH
```

Options:
- --config PATH        Path to YAML or JSON config (mutually exclusive with --init-config)
- --init-config PATH   Write a config template to PATH and exit
- --output-dir PATH    Override the output.directory specified in the config

Notes:
- **Garmin login:** Prefer `GARMIN_EMAIL` and `GARMIN_PASSWORD` in `.env` (same idea as API keys). If unset, the CLI uses `athlete.email` and optional `credentials.password` from the YAML, then prompts for password when still empty.
- `AI_MODE` / `RUN_TYPE` in der `.env` sind für **coach-cli** nicht nötig: `extraction.ai_mode` und `extraction.run_type` in der YAML sind maßgeblich (die CLI schreibt sie vor `reload_config()` in die Umgebung). In der `.env` nur setzen, wenn du **ohne** YAML andere Defaults brauchst (z. B. Skripte, Tests).

## Configuration

Top-level keys:
- athlete: name, email (email optional if `GARMIN_EMAIL` is set in `.env`)
- logging: level (`DEBUG` | `INFO` | `WARNING` | `ERROR`) — steuert den Root-Logger des coach-cli; ohne Eintrag optional `LOG_LEVEL` in `.env`, sonst `INFO`
- context: analysis, planning (freeform text; the AI will follow these constraints)
- extraction: activities_days, metrics_days, ai_mode (`development` < `cost_effective` < `standard` < `gemini_pro` < `openai`; Legacy `pro` = `gemini_pro`)
- competitions: list of {name, date (YYYY-MM-DD), race_type, priority (A/B/C), target_time (HH:MM:SS)}
- output: directory
- credentials: password (optional fallback only; prefer `GARMIN_PASSWORD` in `.env`)

Minimal example:
```yaml
athlete:
  name: "Your Name"
  email: "you@example.com"

context:
  analysis: "Recovering from injury; focus on base building"
  planning: "Olympic triathlon in 12 weeks; build aerobic base"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"

competitions:
  - name: "Target Race"
    date: "2026-04-15"
    race_type: "Olympic"
    priority: "A"

output:
  directory: "./data"

logging:
  level: INFO

credentials:
  password: ""   # optional; prefer GARMIN_PASSWORD in .env
```

Advanced example (derived from real usage):
```yaml
athlete:
  name: "Athlete Name"
  email: "you@example.com"

context:
  analysis: |
    Completed my first 70.3 recently. Great result but exposed durability gaps
    due to last-minute shoe change. Analyze this multisport activity in detail.

  planning: |
    ## Start Date
    Plan should start on **Monday, xxxx-xx-xx**.

    ## Important Needs
    - Functional Strength, Durability & Triathlon Transfer
      Integrate explicit daily micro-workouts (5–10 min).
      Goals: run economy & lower-leg robustness; bike posture & core transfer; durability & recovery.

    - Shoe Adaptation & Running Technique
      Get used to carbon plate shoes (front-foot style) with targeted technique/strength.

    ## Session Constraints (Shoes)
    - Per-session shoe exclusivity: every run is tagged either `carbon` or `non-carbon`.

    ## Training Preferences
    - No indoor bike trainer available.
    - No swimming for now.

    ## Training Zones
    | Discipline | Base Metric                  |
    |------------|------------------------------|
    | Running    | LTHR ≈ 173 bpm / 4:35 min/km |
    | Cycling    | FTP ≈ 271W                   |
    | Heart Rate | Max HR ≈ 193 bpm             |

    ## Closing
    Provide structured daily checklists to support both athletic and personal goals.

extraction:
  activities_days: 21
  metrics_days: 56
  ai_mode: "standard"

competitions:
  - name: "Franklin Meilenlauf"
    date: "2025-10-12"
    race_type: "Half Marathon"
    priority: "A"
    target_time: "01:40:00"

output:
  directory: "./data"

credentials:
  password: ""  # leave empty for secure interactive input
```

Validation tips:
- Date format must be ISO `YYYY-MM-DD` for competitions.
- `athlete.email` is required; the run will fail if missing.

## Outputs

Each run writes into a **new subfolder** under `output.directory` (default: `./data`) to avoid overwriting expensive results:

- `<email>__<ai_mode>__<run_type>__<YYYY-MM-DD>__<HH-MM-SS>/`

Inside that run folder:

- `analysis.html` — training analysis report
- `planning.html` — season overview + compact 4-week plan
- `metrics_expert.json`, `activity_expert.json`, `physiology_expert.json` — structured expert outputs
- `season_plan.md`, `weekly_plan.md` — intermediate planning artifacts
- `summary.json` — metadata and (when available) cost summary:
  - `total_cost_usd` (number or `null`)
  - `total_tokens` (int or `null`)
  - `cost_calculable` (bool)
  - `execution_id`, plus `trace_id` / `root_run_id` when LangSmith is enabled

## Environment

Set at least one provider API key in your environment (e.g., `.env`):
- OPENAI_API_KEY=...
- ANTHROPIC_API_KEY=...
- OPENROUTER_API_KEY=...
- GOOGLE_API_KEY=... (for direct Gemini usage)
- Optional: LANGSMITH_API_KEY=... for observability

The CLI will set `AI_MODE` from your config’s `extraction.ai_mode` (see [`python.run_analysis_from_config()`](../cli/garmin_ai_coach_cli.py:110) where `AI_MODE` is exported at [`os.environ['AI_MODE'] = ai_mode`](../cli/garmin_ai_coach_cli.py:125)).

Provider selection depends on AI mode mapping (cheap/fast → stronger; see [`services/ai/ai_settings.py`](../services/ai/ai_settings.py)):
  - `development` → `gemini-3-flash` (`GOOGLE_API_KEY`)
  - `cost_effective` → `claude-3-haiku` (`ANTHROPIC_API_KEY`)
  - `standard` → `claude-4` (`ANTHROPIC_API_KEY`)
  - `gemini_pro` → `gemini-3.1-pro` (`GOOGLE_API_KEY`); legacy YAML value `pro` is normalized to this
  - `openai` → `gpt-4o` (`OPENAI_API_KEY`)

Fallback routing: if a direct provider key is missing, the app may route supported models through OpenRouter (requires `OPENROUTER_API_KEY`). See [`services/ai/model_config.py`](../services/ai/model_config.py).

Model IDs and providers are declared in [`python.ModelSelector.CONFIGURATIONS`](../services/ai/model_config.py:22), and the provider API key is auto-selected in [`python.ModelSelector.get_llm()`](../services/ai/model_config.py:61).

Practical guidance:
- **Gemini (Flash or Pro):** `GOOGLE_API_KEY` — `development` (Flash) or `gemini_pro` (Pro tier).
- **Anthropic:** `ANTHROPIC_API_KEY` — `cost_effective` (Haiku) or `standard` (Claude 4).
- **OpenAI:** `OPENAI_API_KEY` — `openai` (`gpt-4o`).
- If you want to route via **OpenRouter**, set `OPENROUTER_API_KEY` (this can also act as a fallback router when direct keys are missing for supported models).
