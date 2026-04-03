#!/usr/bin/env python3

import argparse
import asyncio
import getpass
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from core.config import reload_config
from services.ai.ai_settings import AgentRole, ai_settings
from services.ai.langgraph.workflows.planning_workflow import (
    run_complete_analysis_and_planning,
)
from services.ai.utils.plan_storage import FilePlanStorage
from services.garmin import ExtractionConfig, TriathlonCoachDataExtractor
from services.outside.client import OutsideApiGraphQlClient

sys.path.append(str(Path(__file__).parent.parent))


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _parse_log_level(name: str) -> int:
    mapping = logging.getLevelNamesMapping()
    level = mapping.get(name.strip().upper())
    if level is not None:
        return level
    logging.getLogger(__name__).warning("Unbekanntes log level '%s', verwende INFO", name)
    return logging.INFO


def _apply_cli_logging(config_parser: "ConfigParser") -> None:
    """Root-Logger für coach-cli: logging.level aus YAML, sonst LOG_LEVEL aus .env, sonst INFO."""
    raw = config_parser.get_log_level_name()
    if not raw:
        raw = (os.environ.get("LOG_LEVEL") or "").strip()
    if not raw:
        level = logging.INFO
    else:
        level = _parse_log_level(raw)
    logging.basicConfig(format=_LOG_FORMAT, level=level, force=True)
    logging.getLogger(__name__).debug("CLI-Log-Level: %s", logging.getLevelName(level))


logging.basicConfig(format=_LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigParser:

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        content = self.config_path.read_text(encoding="utf-8")

        if self.config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(content)
        elif self.config_path.suffix == ".json":
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")

    def get_athlete_info(self) -> tuple[str, str]:
        name = self.config.get("athlete", {}).get("name", "Athlete")
        email = (os.environ.get("GARMIN_EMAIL") or "").strip()
        if not email:
            email = (self.config.get("athlete", {}).get("email") or "").strip()
        if not email:
            raise ValueError(
                "Garmin-E-Mail fehlt: GARMIN_EMAIL in der .env setzen oder athlete.email in der Config."
            )
        return name, email

    def get_contexts(self) -> tuple[str, str]:
        return (
            self.config.get("context", {}).get("analysis", "").strip(),
            self.config.get("context", {}).get("planning", "").strip()
        )

    def get_extraction_config(self) -> dict[str, Any]:
        return {
            "activities_days": self.config.get("extraction", {}).get("activities_days", 7),
            "metrics_days": self.config.get("extraction", {}).get("metrics_days", 14),
            "ai_mode": self.config.get("extraction", {}).get("ai_mode", "development"),
            "run_type": self.config.get("extraction", {}).get("run_type", "full"),
            "enable_plotting": self.config.get("extraction", {}).get("enable_plotting", False),
            "hitl_enabled": self.config.get("extraction", {}).get("hitl_enabled", True),
            "skip_synthesis": self.config.get("extraction", {}).get("skip_synthesis", False),
        }

    def get_competitions(self) -> list[dict[str, Any]]:
        competitions = self.config.get("competitions", [])
        return [
            {
                "name": comp.get("name", ""),
                "date": comp.get("date", ""),
                "race_type": comp.get("race_type", ""),
                "priority": comp.get("priority", "B"),
                "target_time": comp.get("target_time", ""),
            }
            for comp in competitions
        ]

    def get_output_directory(self) -> Path:
        return Path(self.config.get("output", {}).get("directory", "./data"))

    def get_log_level_name(self) -> str | None:
        raw = self.config.get("logging", {}).get("level")
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    def get_password(self) -> str:
        env_pw = (os.environ.get("GARMIN_PASSWORD") or "").strip()
        if env_pw:
            return env_pw
        yaml_pw = (self.config.get("credentials", {}).get("password") or "").strip()
        if yaml_pw:
            logger.debug(
                "Garmin-Passwort aus der Config-Datei; für getrennte Geheimnisse GARMIN_PASSWORD in .env bevorzugen."
            )
            return yaml_pw
        return getpass.getpass("Enter Garmin Connect password: ")


def fetch_outside_competitions_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    client = OutsideApiGraphQlClient()

    if isinstance(outside_cfg := config.get("outside"), dict) and any(
        isinstance(value, list) for value in outside_cfg.values()
    ):
        return client.get_competitions(outside_cfg)

    aggregate: list[dict[str, Any]] = []

    if isinstance(legacy_bikereg := config.get("bikereg", []), list) and legacy_bikereg:
        aggregate.extend(client.get_competitions(legacy_bikereg))

    if legacy_all := {
        key: entries
        for key in ("runreg", "trireg", "skireg")
        if isinstance(entries := config.get(key, []), list) and entries
    }:
        aggregate.extend(client.get_competitions(legacy_all))

    return aggregate


def _save_html_outputs(output_dir: Path, result: dict[str, Any]) -> list[str]:
    files_generated: list[str] = []

    for filename, key in [
        ("analysis.html", "analysis_html"),
        ("planning.html", "planning_html"),
    ]:
        if content := result.get(key):
            if isinstance(content, dict):
                content = content.get("content", "")

            output_path = output_dir / filename
            output_path.write_text(content, encoding="utf-8")
            files_generated.append(filename)
            logger.info("Saved: %s", output_path)

    return files_generated


def _save_expert_outputs(output_dir: Path, result: dict[str, Any]) -> list[str]:
    files_generated: list[str] = []

    for filename, key in [
        ("metrics_expert.json", "metrics_outputs"),
        ("activity_expert.json", "activity_outputs"),
        ("physiology_expert.json", "physiology_outputs"),
    ]:
        if output := result.get(key):
            output_path = output_dir / filename
            output_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            files_generated.append(filename)
            logger.info("Saved: %s", output_path)

    return files_generated


def _save_plan_outputs(output_dir: Path, result: dict[str, Any]) -> list[str]:
    files_generated: list[str] = []

    storage = FilePlanStorage()
    user_id = result.get("user_id", "cli_user")

    for filename, key in [
        ("season_plan.md", "season_plan"),
        ("weekly_plan.md", "weekly_plan"),
    ]:
        if plan_dict := result.get(key):
            output = plan_dict.get("output", plan_dict) if isinstance(plan_dict, dict) else plan_dict
            if isinstance(output, str):
                output_path = output_dir / filename
                output_path.write_text(output, encoding="utf-8")
                files_generated.append(filename)
                logger.info("Saved: %s", output_path)
                storage.save_plan(user_id, key, output)

    return files_generated


def _safe_folder_component(value: str) -> str:
    """
    Make a string safe for use as a single folder name component on Windows/macOS/Linux.
    Keeps only [A-Za-z0-9_-] and normalizes separators.
    """
    value = value.strip().lower()
    value = value.replace("@", "_").replace(".", "_")
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


async def run_analysis_from_config(config_path: Path) -> None:
    config_parser = ConfigParser(config_path)
    _apply_cli_logging(config_parser)
    athlete_name, email = config_parser.get_athlete_info()
    analysis_context, planning_context = config_parser.get_contexts()
    extraction_settings = config_parser.get_extraction_config()
    _run_type_raw = str(extraction_settings.get("run_type", "full")).lower()
    if _run_type_raw not in ("full", "light"):
        logger.warning(
            "Ungültiger extraction.run_type '%s', verwende 'full'",
            extraction_settings.get("run_type"),
        )
        _run_type_raw = "full"
    extraction_settings["run_type"] = _run_type_raw

    competitions = config_parser.get_competitions()
    outside_competitions = fetch_outside_competitions_from_config(config_parser.config)
    if outside_competitions:
        competitions.extend(outside_competitions)

    base_output_dir = config_parser.get_output_directory()

    logger.info("Starte Analyse für %s", athlete_name)
    logger.info("Ausgabeverzeichnis (Basis): %s", base_output_dir)

    password = config_parser.get_password()

    _ai_raw = str(extraction_settings.get("ai_mode", "development")).strip().lower()
    os.environ["AI_MODE"] = "gemini_pro" if _ai_raw == "pro" else _ai_raw
    os.environ["RUN_TYPE"] = extraction_settings["run_type"]

    # Reload config and settings to pick up the new AI_MODE / RUN_TYPE
    reload_config()
    ai_settings.reload()

    representative_model = ai_settings.get_model_for_role(AgentRole.SUMMARIZER)
    logger.info("AI-Modus: %s (%s)", os.environ["AI_MODE"], representative_model)
    logger.info("Run-Typ: %s (full = Analyse+Planung, light = nur Analyse/analysis.html)", os.environ["RUN_TYPE"])

    now = datetime.now()
    run_folder = (
        f"{_safe_folder_component(email)}__"
        f"{_safe_folder_component(os.environ['AI_MODE'])}__"
        f"{_safe_folder_component(os.environ['RUN_TYPE'])}__"
        f"{now.strftime('%Y-%m-%d')}__"
        f"{now.strftime('%H-%M-%S')}"
    )
    output_dir = base_output_dir / run_folder
    logger.info("Ausgabeverzeichnis (Run): %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Extrahiere Garmin Connect Daten...")
        extractor = TriathlonCoachDataExtractor(email, password)

        extraction_config = ExtractionConfig(
            activities_range=extraction_settings["activities_days"],
            metrics_range=extraction_settings["metrics_days"],
            include_detailed_activities=True,
            include_metrics=True,
        )

        garmin_data = extractor.extract_data(extraction_config)
        logger.info("Datenextraktion abgeschlossen")

        plotting_enabled = extraction_settings.get("enable_plotting", False)
        hitl_enabled = extraction_settings.get("hitl_enabled", True)
        skip_synthesis = extraction_settings.get("skip_synthesis", False)
        run_type = extraction_settings["run_type"]

        logger.info("Plotting aktiviert: %s", plotting_enabled)
        logger.info("HITL aktiviert: %s", hitl_enabled)
        logger.info("Synthese überspringen: %s", skip_synthesis)

        current_date = {"date": now.strftime("%Y-%m-%d"), "day_name": now.strftime("%A")}
        week_dates = [
            {"date": (now + timedelta(days=offset)).strftime("%Y-%m-%d"),
             "day_name": (now + timedelta(days=offset)).strftime("%A")}
            for offset in range(14)
        ]

        logger.info("Starte KI-Analyse und Planung...")

        result = await run_complete_analysis_and_planning(
            user_id="cli_user",
            athlete_name=athlete_name,
            garmin_data=asdict(garmin_data),
            analysis_context=analysis_context,
            planning_context=planning_context,
            competitions=competitions,
            current_date=current_date,
            week_dates=week_dates,
            plotting_enabled=plotting_enabled,
            hitl_enabled=hitl_enabled,
            skip_synthesis=skip_synthesis,
            run_type=run_type,
        )

        logger.info("Speichere Ergebnisse...")

        files_generated: list[str] = []
        files_generated.extend(_save_html_outputs(output_dir, result))
        files_generated.extend(_save_expert_outputs(output_dir, result))
        files_generated.extend(_save_plan_outputs(output_dir, result))

        # Cost reporting: avoid printing misleading "$0.00" when no cost basis is available
        cost_total: float | None = None
        total_tokens: int | None = None

        if isinstance(result.get("cost_summary"), dict) and result["cost_summary"].get("total_cost_usd"):
            cost_total = float(result["cost_summary"]["total_cost_usd"])
            total_tokens = int(result["cost_summary"].get("total_tokens") or 0)
        elif isinstance(result.get("execution_metadata"), dict) and result["execution_metadata"].get("total_cost_usd"):
            cost_total = float(result["execution_metadata"]["total_cost_usd"])
            total_tokens = int(result["execution_metadata"].get("total_tokens") or 0)
        else:
            # Fallback: if we only have local per-agent costs without pricing/usage metadata,
            # we can still report tokens if available, but cost remains "not calculable".
            usage_metadata = result.get("usage_metadata") or {}
            if isinstance(usage_metadata, dict) and usage_metadata:
                # If local cost calculation ran, cost_summary should exist. If not, tokens are still useful.
                token_sum = 0
                for usage in usage_metadata.values():
                    if isinstance(usage, dict):
                        token_sum += int(usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
                total_tokens = token_sum if token_sum > 0 else None

        (output_dir / "summary.json").write_text(
            json.dumps({
                "athlete": athlete_name,
                "analysis_date": datetime.now().isoformat(),
                "competitions": competitions,
                "run_type": run_type,
                "total_cost_usd": cost_total,
                "total_tokens": total_tokens,
                "cost_calculable": cost_total is not None,
                "execution_id": result.get("execution_id", ""),
                "trace_id": result.get("execution_metadata", {}).get("trace_id", ""),
                "root_run_id": result.get("execution_metadata", {}).get("root_run_id", ""),
                "files_generated": files_generated,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        logger.info("✅ Analyse erfolgreich abgeschlossen!")
        if outside_competitions:
            logger.info("✅  %d Outside-Wettkämpfe aus der Konfiguration hinzugefügt", len(outside_competitions))
        logger.info("📁 Ergebnisse gespeichert in: %s", output_dir)
        if cost_total is None:
            token_info = f" ({total_tokens} Tokens)" if isinstance(total_tokens, int) else ""
            logger.info("💰 Gesamtkosten: nicht berechenbar%s", token_info)
        else:
            logger.info("💰 Gesamtkosten: $%.2f (%d Tokens)", cost_total, int(total_tokens or 0))
    except Exception as e:
        logger.error("❌ Analyse fehlgeschlagen: %s", e)
        raise


def create_config_template(output_path: Path) -> None:
    template_path = Path(__file__).parent / "coach_config_template.yaml"

    if template_path.exists():
        output_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("✅ Config template created: %s", output_path)
        logger.info("Edit this file with your settings and run analysis with --config")
    else:
        logger.error("❌ Template file not found")


def main():
    parser = argparse.ArgumentParser(
        description="Garmin AI Coach CLI - AI Triathlon Coach",
        epilog="Example: python garmin_ai_coach_cli.py --config my_config.yaml",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Path to configuration file (YAML or JSON)")
    group.add_argument("--init-config", type=Path, help="Create a configuration template file")

    parser.add_argument("--output-dir", type=Path, help="Override output directory from config")

    args = parser.parse_args()

    if args.init_config:
        create_config_template(args.init_config)
        return

    if args.config:
        try:
            asyncio.run(run_analysis_from_config(args.config))
        except KeyboardInterrupt:
            logger.info("❌ Analysis cancelled by user")
        except Exception as e:
            logger.error("❌ Analysis failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
