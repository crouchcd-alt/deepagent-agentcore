"""
Pydantic models for VCF variant data extraction and validation.

These models are used both locally (double-validation) and inside the AgentCore
sandbox (schema-as-contract pattern). Keep this file self-contained with no
project-local imports so it can be uploaded directly to the sandbox.
"""
# %%

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class VariantRecord(BaseModel):
    """A single validated variant record extracted from a VCF file."""

    chrom: str
    pos: int
    id: Optional[str] = None
    id_type: Optional[str] = None  # e.g. 'COSMIC'
    ref: str
    alt: str
    qual: Optional[float] = None
    filter: Optional[str] = None

    # INFO sub-fields
    gene: Optional[str] = None
    protein: Optional[str] = None
    allele_frequency: Optional[float] = None

    @field_validator("id", mode="before")
    @classmethod
    def coerce_missing_id(cls, v: object) -> Optional[str]:
        """Replace VCF missing-value sentinel '.' with None."""
        if v == "." or v == "":
            return None
        return str(v) if v is not None else None

    @field_validator("qual", mode="before")
    @classmethod
    def coerce_qual(cls, v: object) -> Optional[float]:
        if v is None or v == ".":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("allele_frequency", mode="before")
    @classmethod
    def coerce_allele_frequency(cls, v: object) -> Optional[float]:
        """Coerce allele frequency to a float in [0.0, 1.0]. Accepts '15%' → 0.15."""
        if v is None or v == ".":
            return None
        raw = str(v).strip()
        if raw.endswith("%"):
            try:
                return float(raw[:-1]) / 100.0
            except ValueError:
                return None
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, parsed))

    @field_validator("filter", mode="before")
    @classmethod
    def coerce_missing_filter(cls, v: object) -> Optional[str]:
        if v == ".":
            return None
        return str(v) if v is not None else None

    @model_validator(mode="after")
    def validate_chrom_prefix(self) -> "VariantRecord":
        """Normalize chromosome names — ensure 'chr' prefix."""
        if self.chrom and not self.chrom.startswith("chr"):
            self.chrom = f"chr{self.chrom}"
        return self

    @model_validator(mode="after")
    def validate_id_type(self) -> "VariantRecord":
        if not self.id:
            self.id_type = None
        elif self.id.startswith("COSM"):
            self.id_type = "COSMIC"
        else:
            self.id_type = "unknown"
        return self


class VariantExtractionResult(BaseModel):
    """Container returned by the agent after processing an entire VCF file."""

    records: list[VariantRecord]
    total: int
    validation_errors: list[str] = []
