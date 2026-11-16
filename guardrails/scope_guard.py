"""Scope guard enforcing that each agent may write only its designated fields.

Enforces:
- Extractor: writes `clauses` (`type`, `found`, `text` only; `standard_position`, `risk`, `reason` must be None).
- Retriever: writes `clauses[i].standard_position` only. Cannot alter clause structure, type, found, text, risk, reason.
- Flagger: writes `clauses[i].risk` and `clauses[i].reason` only. Cannot alter clause text, standard_position, etc.
- Summarizer: writes `summary`, `overall_risk`, `recommendation`. Cannot modify `clauses` or `raw_text`.
- No agent may alter `raw_text`.
"""

from __future__ import annotations

from typing import Any, List
from state import ContractState


class ScopeViolationError(PermissionError):
    """Raised when an agent attempts to mutate state fields outside its authorized scope."""
    pass


class ScopeGuard:
    """Validates state transitions between pipeline stages."""

    PERMITTED_TOP_FIELDS = {
        "extractor": {"clauses"},
        "retriever": set(),  # Only allowed to mutate standard_position within existing clauses
        "flagger": set(),    # Only allowed to mutate risk/reason within existing clauses
        "summarizer": {"summary", "overall_risk", "recommendation"},
    }

    @classmethod
    def validate_state_transition(
        cls,
        before: ContractState,
        after: ContractState,
        agent_name: str,
    ) -> None:
        """Verify that after state contains modifications ONLY within the agent's authorized scope."""
        agent = agent_name.lower().strip()
        if agent not in cls.PERMITTED_TOP_FIELDS:
            raise ScopeViolationError(f"Unknown agent '{agent_name}'. Cannot verify scope permission.")

        # 1. raw_text is strictly immutable across all agents
        if before.raw_text != after.raw_text:
            raise ScopeViolationError(f"Agent '{agent}' unauthorized write: modified immutable 'raw_text'.")

        # 2. Extractor scope rules
        if agent == "extractor":
            if after.summary is not None and before.summary is None:
                raise ScopeViolationError("Extractor attempted to set 'summary'.")
            if after.overall_risk is not None and before.overall_risk is None:
                raise ScopeViolationError("Extractor attempted to set 'overall_risk'.")
            if after.recommendation is not None and before.recommendation is None:
                raise ScopeViolationError("Extractor attempted to set 'recommendation'.")
            for i, c in enumerate(after.clauses):
                if c.standard_position is not None:
                    raise ScopeViolationError(f"Extractor attempted to set 'standard_position' on clause {i}.")
                if c.risk is not None:
                    raise ScopeViolationError(f"Extractor attempted to set 'risk' on clause {i}.")
                if c.reason is not None:
                    raise ScopeViolationError(f"Extractor attempted to set 'reason' on clause {i}.")

        # 3. Retriever scope rules
        elif agent == "retriever":
            if before.summary != after.summary:
                raise ScopeViolationError("Retriever attempted to modify 'summary'.")
            if before.overall_risk != after.overall_risk:
                raise ScopeViolationError("Retriever attempted to modify 'overall_risk'.")
            if before.recommendation != after.recommendation:
                raise ScopeViolationError("Retriever attempted to modify 'recommendation'.")
            if len(before.clauses) != len(after.clauses):
                raise ScopeViolationError("Retriever attempted to add/remove clauses.")

            for i, (b_c, a_c) in enumerate(zip(before.clauses, after.clauses)):
                if b_c.type != a_c.type or b_c.found != a_c.found or b_c.text != a_c.text:
                    raise ScopeViolationError(f"Retriever modified core clause fields on clause {i}.")
                if b_c.risk != a_c.risk:
                    raise ScopeViolationError(f"Retriever attempted to modify 'risk' on clause {i}.")
                if b_c.reason != a_c.reason:
                    raise ScopeViolationError(f"Retriever attempted to modify 'reason' on clause {i}.")

        # 4. Flagger scope rules
        elif agent == "flagger":
            if before.summary != after.summary:
                raise ScopeViolationError("Flagger attempted to modify 'summary'.")
            if before.overall_risk != after.overall_risk:
                raise ScopeViolationError("Flagger attempted to modify 'overall_risk'.")
            if before.recommendation != after.recommendation:
                raise ScopeViolationError("Flagger attempted to modify 'recommendation'.")
            if len(before.clauses) != len(after.clauses):
                raise ScopeViolationError("Flagger attempted to add/remove clauses.")

            for i, (b_c, a_c) in enumerate(zip(before.clauses, after.clauses)):
                if b_c.type != a_c.type or b_c.found != a_c.found or b_c.text != a_c.text:
                    raise ScopeViolationError(f"Flagger modified core clause fields on clause {i}.")
                if b_c.standard_position != a_c.standard_position:
                    raise ScopeViolationError(f"Flagger modified 'standard_position' on clause {i}.")

        # 5. Summarizer scope rules
        elif agent == "summarizer":
            if len(before.clauses) != len(after.clauses):
                raise ScopeViolationError("Summarizer attempted to add/remove clauses.")
            for i, (b_c, a_c) in enumerate(zip(before.clauses, after.clauses)):
                if b_c.model_dump() != a_c.model_dump():
                    raise ScopeViolationError(f"Summarizer modified clause data on clause {i}.")
