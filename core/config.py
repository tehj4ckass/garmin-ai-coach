
import logging
import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

env_file = os.getenv("ENV_FILE", ".env")
load_dotenv(env_file)

logger = logging.getLogger(__name__)

_config_cache: "Config | None" = None


class AIMode(Enum):
    """YAML/Env-Werte für extraction.ai_mode — grob: günstig/schnell → teurer/stärker."""

    DEVELOPMENT = "development"
    COST_EFFECTIVE = "cost_effective"
    STANDARD = "standard"
    GEMINI_PRO = "gemini_pro"
    OPENAI = "openai"


class RunType(Enum):
    """Workflow scope: full analysis+planning vs analysis-only (no planning LLM branch)."""

    FULL = "full"
    LIGHT = "light"


@dataclass
class Config:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    google_api_key: str | None = None
    openrouter_api_key: str | None = None

    ai_mode: AIMode = AIMode.STANDARD
    run_type: RunType = RunType.FULL

    @classmethod
    def from_env(cls) -> "Config":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # Coach-CLI setzt AI_MODE vor reload_config() aus coach_config.yaml; .env ist optional (für andere Einstiege).
        ai_mode_str = os.getenv("AI_MODE", "standard").lower().strip()
        if ai_mode_str == "pro":
            ai_mode_str = "gemini_pro"
        try:
            ai_mode = AIMode(ai_mode_str)
        except ValueError:
            ai_mode = AIMode.STANDARD
            logger.info("Warning: Invalid AI_MODE '%s', using %s", ai_mode_str, ai_mode.value)

        run_type_str = os.getenv("RUN_TYPE", "full").lower()
        try:
            run_type = RunType(run_type_str)
        except ValueError:
            run_type = RunType.FULL
            logger.info("Warning: Invalid RUN_TYPE '%s', using %s", run_type_str, run_type.value)

        if anthropic_api_key and not anthropic_api_key.startswith(("sk-ant-api03-", "sk-ant-")):
            raise ValueError("Invalid ANTHROPIC_API_KEY format")

        if openai_api_key and not openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OPENAI_API_KEY format")

        return cls(
            anthropic_api_key=anthropic_api_key,
            ai_mode=ai_mode,
            run_type=run_type,
            openai_api_key=openai_api_key,
            deepseek_api_key=deepseek_api_key,
            google_api_key=google_api_key,
            openrouter_api_key=openrouter_api_key,
        )


def get_config() -> Config:
    global _config_cache  # noqa: PLW0603
    if _config_cache is None:
        _config_cache = Config.from_env()
    return _config_cache


def reload_config() -> Config:
    global _config_cache  # noqa: PLW0603
    _config_cache = None
    return get_config()
