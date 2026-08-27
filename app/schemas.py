from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TriageInput(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class TriageOutput(BaseModel):
    product_area: str
    category: str
    urgency_tier: Literal["P1", "P2", "P3", "P4"]
    urgency_reasoning: str = Field(..., description="Reasoning for the assigned urgency tier and classification.")
    relevant_kb_doc: str | None = None
    routed_team: str
    draft_response: str


class RiskItem(BaseModel):
    issue: str
    quote: str
    ticket_id: str

    @field_validator("ticket_id")
    @classmethod
    def ticket_id_shape(cls, value: str) -> str:
        if not value.startswith("TKT-"):
            raise ValueError("ticket_id must start with TKT-")
        return value


class TAMOutput(BaseModel):
    executive_summary: str = Field(..., min_length=1, max_length=1200)
    open_risks: list[RiskItem]
    talking_points: list[str]

    @field_validator("executive_summary")
    @classmethod
    def summary_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("executive_summary cannot be blank")
        return cleaned


class TAMExtractOutput(BaseModel):
    risks: list[RiskItem] = Field(default_factory=list)


class AccountBriefRequest(BaseModel):
    account_id: str
    as_of: datetime | None = None
