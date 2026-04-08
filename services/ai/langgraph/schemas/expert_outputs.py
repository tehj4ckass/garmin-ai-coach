from pydantic import BaseModel, Field, model_validator

from .agent_outputs import Question


class ReceiverPayload(BaseModel):
    signals: list[str]
    evidence: list[str]
    implications: list[str]
    uncertainty: list[str] | None = None


class ReceiverOutputs(BaseModel):
    for_synthesis: ReceiverPayload = Field(
        ...,
        description="Output für den Synthese-Agenten, der den umfassenden Athletenbericht erstellt"
    )
    for_season_planner: ReceiverPayload = Field(
        ...,
        description="Output für den Saisonplaner, der 12-24 Wochen Makrozyklen entwirft"
    )
    for_weekly_planner: ReceiverPayload = Field(
        ...,
        description="Output für den Wochenplaner, der den Trainingsplan für die nächsten 28 Tage erstellt"
    )


class ExpertOutputBase(BaseModel):
    """
    Expert agents can either:
    - return HITL questions, OR
    - return structured receiver outputs.

    Important: We avoid Union/anyOf here because some providers (e.g. Gemini structured output)
    do not support JSON Schema `anyOf`.
    """

    questions: list[Question] | None = Field(
        default=None,
        description="Optional: HITL questions. If set, outputs/content must be omitted.",
    )
    outputs: ReceiverOutputs | None = Field(
        default=None,
        description="Optional: structured outputs for downstream consumers. If set, questions must be omitted.",
    )

    @model_validator(mode="after")
    def _validate_exactly_one_mode(self):
        has_questions = bool(self.questions)
        has_outputs = self.outputs is not None
        if has_questions == has_outputs:
            raise ValueError("Provide exactly one of 'questions' or 'outputs'.")
        return self


class MetricsExpertOutputs(ExpertOutputBase):
    pass


class ActivityExpertOutputs(ExpertOutputBase):
    pass


class PhysiologyExpertOutputs(ExpertOutputBase):
    pass
