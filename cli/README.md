# Garmin AI Coach CLI (primary interface)

Command-line interface for the AI endurance coach: YAML/JSON config, Garmin extraction, LangGraph multi-agent analysis, and HTML reports.

**This fork:** agent prompts and **report text default to German** (see [About this fork](../README.md#about-this-fork) in the root README). It also adds tiered **`extraction.ai_mode`** (Google / Anthropic / OpenAI) and **`extraction.run_type`** (`full` = analysis + planning, `light` = `analysis.html` only).

- CLI script: [garmin_ai_coach_cli.py](garmin_ai_coach_cli.py)
- **Coach Chat** (*talk to your training data*, Chainlit UI): [qa_chainlit_app.py](qa_chainlit_app.py) — `pixi run qa-chat`
- Config: `pixi run coach-init my_config.yaml` or start from [coach_config.yaml](../coach_config.yaml) / [cli/coach_config.yaml](coach_config.yaml)
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

### Post-run Q&A (Chainlit) — *Coach Chat*

After a successful run, artifacts live under `output.directory` (default `./data/<run-folder>/`). **Talk to your training data** in a browser chat (reads `coach_config.yaml` by default for **`extraction.ai_mode`** / API keys — unless you override with **`QA_AI_MODE`**).

```bash
# from repo root
pixi run qa-chat
```

On first message, pick a run by **number**, **folder name**, or **full path** to a directory that contains `garmin_data.json` and/or `*_expert.json`.

Optional env: `COACH_CONFIG` (path to YAML, default `coach_config.yaml`), `GARMIN_COACH_DATA` or `COACH_DATA_DIR` to override the data root without editing YAML.

**Prompt cost:** Coach Chat sends large artifacts (expert JSON, `garmin_data`, HTML). Input tokens dominate the bill. Defaults now use a **`balanced`** context profile (smaller caps than before). Set **`QA_CONTEXT_PROFILE=full`** to restore the previous (very large) context, or **`economy`** for minimum size.

| Variable | Default | Meaning |
|----------|---------|---------|
| `QA_AI_MODE` | _(unset)_ | Optional: same values as `extraction.ai_mode` (`development`, `cost_effective`, `standard`, `gemini_pro` / `pro`, `openai`). Overrides YAML **for this Chainlit app only** so you can keep `standard` for the CLI but e.g. `cost_effective` or `development` for chat. |
| `QA_CONTEXT_PROFILE` | `balanced` | `balanced` (default) · `economy` (smaller) · `full` (legacy caps). Fine-tune with `QA_CONTEXT_*_MAX_CHARS` below. |
| `QA_CONTEXT_SUMMARY_MAX_CHARS` | _(profile)_ | Override max characters embedded from `summary.json`. |
| `QA_CONTEXT_EXPERT_MAX_CHARS` | _(profile)_ | Per expert JSON (`metrics` / `activity` / `physiology`). |
| `QA_CONTEXT_GARMIN_MAX_CHARS` | _(profile)_ | `garmin_data.json` excerpt. |
| `QA_CONTEXT_MARKDOWN_MAX_CHARS` | _(profile)_ | `season_plan.md` / `weekly_plan.md` each. |
| `QA_CONTEXT_HTML_MAX_CHARS` | _(profile)_ | `analysis.html` excerpt. |
| `QA_STREAM_TIMEOUT_SEC` | `60` | Max seconds per assistant reply (wall clock); then stream aborts with a footer. |
| `QA_RESPONSE_BUDGET` | `medium` | If `QA_MAX_OUTPUT_TOKENS` is unset: `short` (~896) · `medium` (~1536, default) · `long` (~3072) · `full` (4096). Shorter replies = fewer completion tokens (lower $). |
| `QA_MAX_OUTPUT_TOKENS` | _(see budget)_ | Explicit cap overrides `QA_RESPONSE_BUDGET`. LangChain `.bind(max_tokens=…)` (best-effort per provider). |
| `QA_MAX_USER_MESSAGE_CHARS` | `12000` | Reject oversized user paste to protect context + cost. |

While a reply is streaming, use the **Stop** control in the Chainlit UI; it sets a cancel flag between chunks.

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
- **`AI_MODE` / `RUN_TYPE` in `.env`:** not required for coach-cli — `extraction.ai_mode` and `extraction.run_type` in YAML win (the CLI exports them before `reload_config()`). Set env vars only for non-CLI entrypoints (e.g. tests, scripts).

## Configuration

Top-level keys:
- athlete: name, email (email optional if `GARMIN_EMAIL` is set in `.env`)
- logging: level (`DEBUG` | `INFO` | `WARNING` | `ERROR`) — root logger for coach-cli; if omitted, optional `LOG_LEVEL` in `.env`, else `INFO`
- context: analysis, planning (freeform text; the AI will follow these constraints)
- extraction: activities_days, metrics_days, ai_mode (`development` < `cost_effective` < `standard` < `gemini_pro` < `openai`; legacy `pro` → `gemini_pro`), run_type (`full` | `light` — `light` skips the whole planning branch; only `analysis.html`)
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
  run_type: "full"

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
  run_type: "full"

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
- Garmin email: either `athlete.email` in YAML or `GARMIN_EMAIL` in `.env` (see Notes above).

## Outputs

Each run writes into a **new subfolder** under `output.directory` (default: `./data`) to avoid overwriting expensive results:

- `<email>__<ai_mode>__<run_type>__<YYYY-MM-DD>__<HH-MM-SS>/`

Inside that run folder:

- `analysis.html` — training analysis report (**German** copy in this fork)
- `planning.html` — season overview + compact 4-week plan (**not** produced for `run_type: light` in most runs)
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
