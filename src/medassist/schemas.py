"""Pydantic schemas used for structured LLM output."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProfileUpdate(BaseModel):
    """Patient information extracted from the conversation (null = not mentioned)."""

    age: Optional[int] = Field(None, description="Age in years")
    sex: Optional[str] = Field(None, description="'male', 'female' or 'other'")
    symptoms: Optional[str] = Field(None, description="Short description of symptoms")
    allergies: Optional[str] = Field(
        None, description="Known allergies. Write 'none' if the user says they have none"
    )
    current_medications: Optional[str] = Field(
        None, description="Medicines currently taken. Write 'none' if the user takes none"
    )


class RouteDecision(BaseModel):
    """Decision of the orchestrator agent."""

    route: Literal["disease", "medicine", "both", "direct", "emergency"]
    reasoning: str = Field("", description="One short sentence justifying the route")
