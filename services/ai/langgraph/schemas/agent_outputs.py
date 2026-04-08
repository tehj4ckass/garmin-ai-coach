from pydantic import BaseModel, Field, model_validator


class Question(BaseModel):
    id: str = Field(..., description="Unique identifier (e.g., 'metrics_q1')")
    message: str = Field(..., description="Question text")
    context: str | None = Field(None, description="Additional context")
    message_type: str = Field("question", description="Type of message")


class AgentOutput(BaseModel):
    """
    Agent produces either HITL questions or plain content.

    Important: Avoid Union/anyOf in the JSON schema for provider compatibility.
    """

    questions: list[Question] | None = Field(
        default=None,
        description="Optional: HITL questions. If set, content must be omitted.",
    )
    content: str | None = Field(
        default=None,
        description="Optional: downstream content. If set, questions must be omitted.",
    )

    @model_validator(mode="after")
    def _validate_exactly_one_mode(self):
        has_questions = bool(self.questions)
        has_content = self.content is not None and self.content != ""
        if has_questions == has_content:
            raise ValueError("Provide exactly one of 'questions' or 'content'.")
        return self
