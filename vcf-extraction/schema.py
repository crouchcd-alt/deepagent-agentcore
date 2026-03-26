"""
Pydantic models for VCF variant data extraction and validation.

These models represent structured variant records parsed from a VCF file.
They are designed to be used both locally and inside the AgentCore sandbox.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class VariantRecord(BaseModel):
    """
    A single validated variant record extracted from a VCF file.

    Fields map directly to standard VCF columns plus INFO sub-fields
    for gene annotation, protein change, and allele frequency.
    """

    chrom: str
    pos: int
    id: Optional[str] = None
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
        """
        Coerce QUAL field to float.

        Handles the VCF missing-value sentinel '.' and string representations.
        """
        if v is None or v == ".":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("allele_frequency", mode="before")
    @classmethod
    def coerce_allele_frequency(cls, v: object) -> Optional[float]:
        """
        Coerce allele frequency to a float in [0.0, 1.0].

        Accepts percentage strings (e.g. '15%' → 0.15) and numeric strings.
        Values outside [0.0, 1.0] are clamped with a warning rather than
        raising an error, so partial records are preserved.
        """
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
        # Clamp to valid range
        return max(0.0, min(1.0, parsed))

    @field_validator("filter", mode="before")
    @classmethod
    def coerce_missing_filter(cls, v: object) -> Optional[str]:
        """Replace VCF missing-value sentinel '.' with None."""
        if v == ".":
            return None
        return str(v) if v is not None else None

    @model_validator(mode="after")
    def validate_chrom_prefix(self) -> "VariantRecord":
        """
        Normalise chromosome names.

        Ensures the chromosome string starts with 'chr'. VCF files from
        different sources may omit the prefix (e.g. '1' vs 'chr1').
        """
        if self.chrom and not self.chrom.startswith("chr"):
            self.chrom = f"chr{self.chrom}"
        return self


class VariantExtractionResult(BaseModel):
    """
    Container returned by the agent after processing an entire VCF file.

    Attributes:
        records:          List of successfully validated variant records.
        total:            Total number of data rows processed (excludes header lines).
        validation_errors: Human-readable descriptions of any rows that could not
                           be parsed or validated.
    """

    records: list[VariantRecord]
    total: int
    validation_errors: list[str] = []
