from dataclasses import dataclass, field
from enum import Enum

from core.config import AIMode, get_config


class AgentRole(Enum):
    SUMMARIZER = "summarizer"
    METRICS_EXPERT = "metrics_expert"
    PHYSIOLOGY_EXPERT = "physiology_expert"
    ACTIVITY_EXPERT = "activity_expert"
    SYNTHESIS = "synthesis"
    WORKOUT = "workout"
    SEASON_PLANNER = "season_planner"
    FORMATTER = "formatter"


@dataclass
class AISettings:
    mode: AIMode

    model_assignments: dict[AIMode, dict[AgentRole, str]] = field(
        default_factory=lambda: {
            AIMode.DEVELOPMENT: {
                AgentRole.SUMMARIZER: "gemini-3-flash",
                AgentRole.FORMATTER: "gemini-3-flash",
                AgentRole.METRICS_EXPERT: "gemini-3-flash",
                AgentRole.PHYSIOLOGY_EXPERT: "gemini-3-flash",
                AgentRole.ACTIVITY_EXPERT: "gemini-3-flash",
                AgentRole.SYNTHESIS: "gemini-3-flash",
                AgentRole.WORKOUT: "gemini-3-flash",
                AgentRole.SEASON_PLANNER: "gemini-3-flash",
            },
            AIMode.COST_EFFECTIVE: {
                AgentRole.SUMMARIZER: "claude-3-haiku",
                AgentRole.FORMATTER: "claude-3-haiku",
                AgentRole.METRICS_EXPERT: "claude-3-haiku",
                AgentRole.PHYSIOLOGY_EXPERT: "claude-3-haiku",
                AgentRole.ACTIVITY_EXPERT: "claude-3-haiku",
                AgentRole.SYNTHESIS: "claude-3-haiku",
                AgentRole.WORKOUT: "claude-3-haiku",
                AgentRole.SEASON_PLANNER: "claude-3-haiku",
            },
            AIMode.STANDARD: {
                AgentRole.SUMMARIZER: "claude-4",
                AgentRole.FORMATTER: "claude-4",
                AgentRole.METRICS_EXPERT: "claude-4",
                AgentRole.PHYSIOLOGY_EXPERT: "claude-4",
                AgentRole.ACTIVITY_EXPERT: "claude-4",
                AgentRole.SYNTHESIS: "claude-4",
                AgentRole.WORKOUT: "claude-4",
                AgentRole.SEASON_PLANNER: "claude-4",
            },
            AIMode.GEMINI_PRO: {
                AgentRole.SUMMARIZER: "gemini-3.1-pro",
                AgentRole.FORMATTER: "gemini-3.1-pro",
                AgentRole.METRICS_EXPERT: "gemini-3.1-pro",
                AgentRole.PHYSIOLOGY_EXPERT: "gemini-3.1-pro",
                AgentRole.ACTIVITY_EXPERT: "gemini-3.1-pro",
                AgentRole.SYNTHESIS: "gemini-3.1-pro",
                AgentRole.WORKOUT: "gemini-3.1-pro",
                AgentRole.SEASON_PLANNER: "gemini-3.1-pro",
            },
            AIMode.OPENAI: {
                AgentRole.SUMMARIZER: "gpt-4o",
                AgentRole.FORMATTER: "gpt-4o",
                AgentRole.METRICS_EXPERT: "gpt-4o",
                AgentRole.PHYSIOLOGY_EXPERT: "gpt-4o",
                AgentRole.ACTIVITY_EXPERT: "gpt-4o",
                AgentRole.SYNTHESIS: "gpt-4o",
                AgentRole.WORKOUT: "gpt-4o",
                AgentRole.SEASON_PLANNER: "gpt-4o",
            },
        }
    )

    def get_model_for_role(self, role: AgentRole) -> str:
        return self.model_assignments[self.mode][role]

    @classmethod
    def load_settings(cls) -> "AISettings":
        return cls(mode=get_config().ai_mode)

    def reload(self) -> None:
        self.mode = get_config().ai_mode


# Global settings instance
ai_settings = AISettings.load_settings()
