from pydantic import BaseModel, Field

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
    output: list[Question] | ReceiverOutputs = Field(
        ...,
        description="ENTWEDER Fragen für HITL ODER vollständiger Output für nachgelagerte Konsumenten"
    )


class MetricsExpertOutputs(ExpertOutputBase):
    pass


class ActivityExpertOutputs(ExpertOutputBase):
    pass


class PhysiologyExpertOutputs(ExpertOutputBase):
    pass
