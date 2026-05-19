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
                AgentRole.SUMMARIZER: "gemini-3.5-flash",
                AgentRole.FORMATTER: "gemini-3.5-flash",
                AgentRole.METRICS_EXPERT: "gemini-3.5-flash",
                AgentRole.PHYSIOLOGY_EXPERT: "gemini-3.5-flash",
                AgentRole.ACTIVITY_EXPERT: "gemini-3.5-flash",
                AgentRole.SYNTHESIS: "gemini-3.5-flash",
                AgentRole.WORKOUT: "gemini-3.5-flash",
                AgentRole.SEASON_PLANNER: "gemini-3.5-flash",
            },
            AIMode.COST_EFFECTIVE: {
                AgentRole.SUMMARIZER: "gemini-3.1-flash-lite",
                AgentRole.FORMATTER: "gemini-3.1-flash-lite",
                AgentRole.METRICS_EXPERT: "gemini-3.1-flash-lite",
                AgentRole.PHYSIOLOGY_EXPERT: "gemini-3.1-flash-lite",
                AgentRole.ACTIVITY_EXPERT: "gemini-3.1-flash-lite",
                AgentRole.SYNTHESIS: "gemini-3.1-flash-lite",
                AgentRole.WORKOUT: "gemini-3.1-flash-lite",
                AgentRole.SEASON_PLANNER: "gemini-3.1-flash-lite",
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
            AIMode.PRO: {
                AgentRole.SUMMARIZER: "claude-opus-4.7",
                AgentRole.FORMATTER: "claude-opus-4.7",
                AgentRole.METRICS_EXPERT: "claude-opus-4.7",
                AgentRole.PHYSIOLOGY_EXPERT: "claude-opus-4.7",
                AgentRole.ACTIVITY_EXPERT: "claude-opus-4.7",
                AgentRole.SYNTHESIS: "claude-opus-4.7",
                AgentRole.WORKOUT: "claude-opus-4.7",
                AgentRole.SEASON_PLANNER: "claude-opus-4.7",
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
