"""Output guardrails for contract review pipeline.

Enforces:
- Pydantic schema validation per agent.
- Strict enum checks on `risk` and clause `type`.
- `reason` non-empty and <= 300 characters.
- Anti-hallucination check: Extractor clause text must be a verbatim substring of raw_text.
- Completeness check: Summarizer must mention every flagged clause by name (< 200 words).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from state import Clause, ClauseRisk, ClauseType


class OutputValidationError(ValueError):
    """Raised when agent output fails schema, enum, or semantic constraints."""
    pass


class ExtractorOutputSchema(BaseModel):
    """Schema for validating Extractor JSON output."""
    class ExtractedClauseItem(BaseModel):
        type: ClauseType
        found: bool = True
        text: str = ""

    clauses: List[ExtractedClauseItem] = Field(default_factory=list)


class FlaggerOutputSchema(BaseModel):
    """Schema for validating Flagger JSON output."""
    risk: ClauseRisk
    reason: str = Field(min_length=1, max_length=300)


def _normalize_spaces(text: str) -> str:
    """Normalize consecutive whitespace characters to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def validate_extractor_output(output_raw: Any, raw_contract_text: str) -> List[Clause]:
    """Validate Extractor output against schema and enforce anti-hallucination substring rule.
    
    Args:
        output_raw: Raw JSON string or parsed dict from Extractor.
        raw_contract_text: The source contract text.
        
    Returns:
        List of validated Clause models.
        
    Raises:
        OutputValidationError: If schema is invalid or extracted text is not a substring.
    """
    if isinstance(output_raw, str):
        try:
            # Strip markdown code fence if present
            cleaned = output_raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
        except Exception as e:
            raise OutputValidationError(f"Extractor output is not valid JSON: {e}")
    elif isinstance(output_raw, dict):
        parsed = output_raw
    else:
        raise OutputValidationError(f"Expected dict or JSON string for Extractor output, got {type(output_raw).__name__}")

    try:
        validated_schema = ExtractorOutputSchema.model_validate(parsed)
    except ValidationError as e:
        raise OutputValidationError(f"Extractor schema validation failed: {e}")

    normalized_raw = _normalize_spaces(raw_contract_text)
    validated_clauses: List[Clause] = []

    for item in validated_schema.clauses:
        clause_text = item.text.strip()
        if item.found:
            if not clause_text:
                raise OutputValidationError(
                    f"Clause of type '{item.type.value}' was marked found=True but returned empty text."
                )
            # Anti-hallucination check: must be a substring of raw_contract_text
            # Check both verbatim and whitespace-normalized
            if clause_text not in raw_contract_text and _normalize_spaces(clause_text) not in normalized_raw:
                raise OutputValidationError(
                    f"Anti-hallucination violation: extracted text for '{item.type.value}' is not a verbatim substring of contract text."
                )
        else:
            clause_text = ""

        validated_clauses.append(
            Clause(
                type=item.type,
                found=item.found,
                text=clause_text,
            )
        )

    return validated_clauses


def validate_flagger_output(output_raw: Any) -> Tuple[ClauseRisk, str]:
    """Validate Flagger output against schema, enum check, and reason length constraint (<=300 chars).
    
    Args:
        output_raw: Raw JSON string or parsed dict from Flagger.
        
    Returns:
        Tuple of (ClauseRisk, reason).
        
    Raises:
        OutputValidationError: If JSON is invalid, risk enum is unknown, or reason violates constraints.
    """
    if isinstance(output_raw, str):
        try:
            cleaned = output_raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
        except Exception as e:
            raise OutputValidationError(f"Flagger output is not valid JSON: {e}")
    elif isinstance(output_raw, dict):
        parsed = output_raw
    else:
        raise OutputValidationError(f"Expected dict or JSON string for Flagger output, got {type(output_raw).__name__}")

    try:
        validated = FlaggerOutputSchema.model_validate(parsed)
    except ValidationError as e:
        raise OutputValidationError(f"Flagger schema validation failed: {e}")

    # Enforce non-empty and <=300 characters explicitly
    reason_clean = validated.reason.strip()
    if not reason_clean:
        raise OutputValidationError("Flagger reason cannot be empty or whitespace only.")
    if len(reason_clean) > 300:
        raise OutputValidationError(f"Flagger reason length ({len(reason_clean)}) exceeds 300 characters limit.")

    return validated.risk, reason_clean


def validate_summarizer_output(summary_text: str, clauses: List[Clause]) -> str:
    """Validate Summarizer output: must be < 200 words and mention every flagged clause by name.
    
    Args:
        summary_text: The plain text summary generated by the Summarizer agent.
        clauses: List of evaluated clauses to verify flagged presence.
        
    Returns:
        The validated summary text.
        
    Raises:
        OutputValidationError: If word count exceeds 200 words or a flagged clause name is missing.
    """
    if not isinstance(summary_text, str) or not summary_text.strip():
        raise OutputValidationError("Summary text is empty or not a valid string.")

    # Word count check (< 200 words)
    words = summary_text.strip().split()
    if len(words) >= 200:
        raise OutputValidationError(f"Summary exceeds length limit: {len(words)} words (maximum 199 words allowed).")

    # Flagged clauses check: every REJECT or REVIEW clause MUST appear by name
    summary_lower = summary_text.lower()
    flagged = [c for c in clauses if c.risk in (ClauseRisk.REJECT, ClauseRisk.REVIEW) or not c.found]

    for c in flagged:
        c_name = c.type.value.lower()
        if c_name not in summary_lower:
            raise OutputValidationError(
                f"Summary completeness violation: flagged clause '{c_name}' ({c.risk.value if c.risk else 'NOT FOUND'}) "
                f"was not mentioned in the summary."
            )

    return summary_text.strip()
