"""Shared State definitions for Legal Contract Review and Risk Flagging System."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    """Supported clause types to extract and evaluate."""
    LIABILITY = "liability"
    TERMINATION = "termination"
    PAYMENT = "payment"
    CONFIDENTIALITY = "confidentiality"


class ClauseRisk(str, Enum):
    """Clause-level risk evaluation labels.
    
    Hard requirement: Keep ClauseRisk (ACCEPTABLE / REVIEW / REJECT)
    separate from overall Recommendation (ACCEPT / NEGOTIATE / REJECT).
    """
    ACCEPTABLE = "ACCEPTABLE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class OverallRisk(str, Enum):
    """Overall contract risk level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(str, Enum):
    """Overall contract recommendation labels."""
    ACCEPT = "ACCEPT"
    NEGOTIATE = "NEGOTIATE"
    REJECT = "REJECT"


# Named constants for recommendation thresholds (§4 Hard Requirement)
MAX_REVIEW_FOR_ACCEPT: int = 1
MIN_REVIEW_FOR_NEGOTIATE: int = 2


class Clause(BaseModel):
    """Individual extracted clause and its risk evaluation."""
    type: ClauseType = Field(description="Clause category (liability, termination, payment, confidentiality)")
    found: bool = Field(default=True, description="Whether the clause was found in the contract")
    text: str = Field(default="", description="Verbatim clause text from contract (empty if not found)")
    standard_position: Optional[str] = Field(default=None, description="Firm's standard position retrieved from policy library")
    risk: Optional[ClauseRisk] = Field(default=None, description="Clause-level risk classification")
    reason: Optional[str] = Field(default=None, description="Explanation for the risk classification (max 300 chars)")


class ContractState(BaseModel):
    """Shared state passed sequentially through the agent pipeline."""
    raw_text: str = Field(description="Original, un-sanitized or sanitized contract text")
    clauses: List[Clause] = Field(default_factory=list, description="Extracted clauses and evaluations")
    summary: Optional[str] = Field(default=None, description="Executive summary for reviewer (<200 words)")
    overall_risk: Optional[OverallRisk] = Field(default=None, description="Overall risk rating (LOW, MEDIUM, HIGH)")
    recommendation: Optional[Recommendation] = Field(default=None, description="Overall recommendation (ACCEPT, NEGOTIATE, REJECT)")
    audit_log: List[Dict[str, Any]] = Field(default_factory=list, description="Audit trail of agent executions")


def compute_overall_recommendation(clauses: List[Clause]) -> tuple[OverallRisk, Recommendation]:
    """Compute overall contract risk and recommendation based on clause risks in code.
    
    Rule:
    - Any REJECT -> REJECT / HIGH
    - Else 2+ REVIEW -> NEGOTIATE / MEDIUM
    - Else -> ACCEPT / LOW
    """
    has_reject = any(c.risk == ClauseRisk.REJECT for c in clauses)
    review_count = sum(1 for c in clauses if c.risk == ClauseRisk.REVIEW)

    if has_reject:
        return OverallRisk.HIGH, Recommendation.REJECT
    elif review_count >= MIN_REVIEW_FOR_NEGOTIATE:
        return OverallRisk.MEDIUM, Recommendation.NEGOTIATE
    else:
        return OverallRisk.LOW, Recommendation.ACCEPT
