"""
Pydantic models for pathology report data extraction and validation.

These models are used both locally (double-validation) and inside the AgentCore
sandbox (schema-as-contract pattern). Keep this file self-contained with no
project-local imports so it can be uploaded directly to the sandbox.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator


class Citation(BaseModel):
    """A verifiable reference back to the source PDF."""

    page: int
    text: str  # exact quote from the PDF that supports the extracted value

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, v: object) -> str:
        return str(v).strip()


class CitedField(BaseModel):
    """An extracted value paired with its source citation."""

    value: str
    citation: Citation


class PathologyExtractionResult(BaseModel):
    """Key clinical fields extracted from a pathology report."""

    age: Optional[CitedField] = None
    primary_diagnosis: CitedField
    performance_status: Optional[CitedField] = None

    @field_validator("performance_status", "age", mode="before")
    @classmethod
    def empty_to_none(cls, v: object) -> Optional[CitedField]:
        if v is None:
            return None
        if isinstance(v, dict) and (
            v.get("value") is None or str(v.get("value", "")).strip() == ""
        ):
            return None
        return v
