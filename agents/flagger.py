"""Risk Flagger Agent.

Compares contract clauses against firm's standard policy positions.
Enforces:
- Schema validation and strict enum checking (ACCEPTABLE, REVIEW, REJECT).
- Reason length constraint (non-empty, <= 300 chars).
- Code-enforced REVIEW when found=False (no LLM call).
- Uncertainty rule: defaults to REVIEW, never ACCEPTABLE.
- Escalation rule: validation failure escalates to REVIEW.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from config.agent_config import build_agent_kernel, get_agent_config, token_tracker
from guardrails.output_guard import OutputValidationError, validate_flagger_output
from state import Clause, ClauseRisk, ContractState

logger = logging.getLogger("contract_review.agents.flagger")


class RiskFlaggerAgent:
    """Agent responsible for classifying clause risk against benchmark standards."""

    def __init__(self, mock_mode: bool = False):
        self.agent_name = "flagger"
        self.mock_mode = mock_mode
        self.config = get_agent_config(self.agent_name)
        self.kernel, self.chat_service = build_agent_kernel(self.agent_name, mock_mode=mock_mode)
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "flagger.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    async def _evaluate_single_clause(self, clause: Clause) -> Tuple[ClauseRisk, str, str]:
        """Evaluate one clause against its standard position.
        
        Returns:
            (risk, reason, validation_status)
        """
        # Hard requirement: If found=false -> force REVIEW in code, no LLM call (§4)
        if not clause.found:
            return (
                ClauseRisk.REVIEW,
                "Required clause is absent from contract text; flagged for manual review.",
                "forced_code_review",
            )

        # If no standard policy was found above threshold -> REVIEW in code
        if not clause.standard_position or clause.standard_position == "no policy found":
            return (
                ClauseRisk.REVIEW,
                "No standard firm policy benchmark found for this clause; manual review required.",
                "no_policy_fallback",
            )

        user_content = (
            f"Clause Type: {clause.type.value}\n"
            f"Clause Text:\n<contract_clause>\n{clause.text}\n</contract_clause>\n\n"
            f"Firm Standard Position:\n{clause.standard_position}"
        )

        chat_history = ChatHistory()
        chat_history.add_system_message(self.system_prompt)
        chat_history.add_user_message(user_content)

        settings = OpenAIChatPromptExecutionSettings(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"} if not self.mock_mode else None,
        )

        try:
            # First attempt
            responses = await self.chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=settings,
            )
            raw_content = responses[0].content or ""
            risk, reason = validate_flagger_output(raw_content)
            return risk, reason, "passed"

        except (OutputValidationError, Exception) as err:
            logger.warning("Flagger first attempt failed for %s (%s). Retrying once.", clause.type.value, err)
            try:
                # Retry once with validation error feedback
                retry_history = ChatHistory()
                retry_history.add_system_message(self.system_prompt)
                retry_history.add_user_message(user_content)
                retry_history.add_assistant_message(raw_content if 'raw_content' in locals() else "")
                retry_history.add_user_message(
                    f"Your previous response failed validation: {err}. "
                    'Return valid JSON only: {"risk": "ACCEPTABLE|REVIEW|REJECT", "reason": "<max 300 chars>"}.'
                )

                retry_responses = await self.chat_service.get_chat_message_contents(
                    chat_history=retry_history,
                    settings=settings,
                )
                retry_content = retry_responses[0].content or ""
                risk, reason = validate_flagger_output(retry_content)
                return risk, reason, "retried_and_passed"

            except (OutputValidationError, Exception) as retry_err:
                # Escalation rule: failures always escalate risk to REVIEW, never downgrade
                logger.error("Flagger retry failed for %s (%s). Escalating to REVIEW.", clause.type.value, retry_err)
                return (
                    ClauseRisk.REVIEW,
                    "Automated risk evaluation failed validation; escalated to manual review.",
                    "fallback_used",
                )

    async def execute(self, state: ContractState) -> Tuple[List[Clause], Dict[str, Any]]:
        """Evaluate risk for all clauses and return updated clauses + audit log."""
        start_time = time.time()
        input_data = "".join(f"{c.type.value}:{c.text}:{c.standard_position}" for c in state.clauses)
        input_hash = hashlib.sha256(input_data.encode("utf-8")).hexdigest()

        updated_clauses: List[Clause] = []
        statuses = []

        for clause in state.clauses:
            clause_copy = clause.model_copy()
            risk, reason, status = await self._evaluate_single_clause(clause_copy)
            clause_copy.risk = risk
            clause_copy.reason = reason
            updated_clauses.append(clause_copy)
            statuses.append(status)

        latency = time.time() - start_time
        token_info = token_tracker.get_usage(self.agent_name)

        overall_status = "passed"
        if "fallback_used" in statuses:
            overall_status = "fallback_used"
        elif "retried_and_passed" in statuses:
            overall_status = "retried_and_passed"

        audit_entry = {
            "agent": self.agent_name,
            "input_hash": input_hash,
            "output": [
                {"type": c.type.value, "risk": c.risk.value if c.risk else None, "reason": c.reason}
                for c in updated_clauses
            ],
            "validation_result": overall_status,
            "tokens": token_info,
            "latency": round(latency, 4),
        }

        return updated_clauses, audit_entry
