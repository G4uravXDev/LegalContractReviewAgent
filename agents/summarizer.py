"""Summarizer Agent.

Generates a concise reviewer-facing summary (<200 words) listing all REJECT and REVIEW
clauses first, then acceptable clauses in one line.
Computes overall recommendation strictly in code:
- Any REJECT -> REJECT / HIGH
- Else 2+ REVIEW -> NEGOTIATE / MEDIUM
- Else -> ACCEPT / LOW
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from config.agent_config import build_agent_kernel, get_agent_config, token_tracker
from guardrails.output_guard import OutputValidationError, validate_summarizer_output
from state import (
    Clause,
    ClauseRisk,
    ContractState,
    OverallRisk,
    Recommendation,
    compute_overall_recommendation,
)

logger = logging.getLogger("contract_review.agents.summarizer")


class ContractSummarizerAgent:
    """Agent responsible for executive review summaries and code-calculated recommendations."""

    def __init__(self, mock_mode: bool = False):
        self.agent_name = "summarizer"
        self.mock_mode = mock_mode
        self.config = get_agent_config(self.agent_name)
        self.kernel, self.chat_service = build_agent_kernel(self.agent_name, mock_mode=mock_mode)
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "summarizer.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _deterministic_template_fallback(self, clauses: List[Clause]) -> str:
        """Deterministic fallback summary guaranteeing <200 words and naming all flagged clauses."""
        rejects = [c for c in clauses if c.risk == ClauseRisk.REJECT]
        reviews = [c for c in clauses if c.risk == ClauseRisk.REVIEW or not c.found]
        acceptable = [c for c in clauses if c.risk == ClauseRisk.ACCEPTABLE and c.found]

        lines = ["Contract Review Summary:"]

        if rejects:
            lines.append("Critical Rejections:")
            for c in rejects:
                lines.append(f"- {c.type.value.upper()} (REJECT): {c.reason or 'Unacceptable policy deviation.'}")

        if reviews:
            lines.append("Clauses Requiring Review/Negotiation:")
            for c in reviews:
                lines.append(f"- {c.type.value.upper()} (REVIEW): {c.reason or 'Deviates from standard position.'}")

        if not rejects and not reviews:
            lines.append("All analyzed provisions fully adhere to the standard policy positions.")

        if acceptable:
            acc_names = ", ".join(c.type.value.title() for c in acceptable)
            lines.append(f"Acceptable terms: {acc_names} comply with standard firm positions.")

        return "\n".join(lines)

    async def execute(
        self,
        state: ContractState,
    ) -> Tuple[str, OverallRisk, Recommendation, Dict[str, Any]]:
        """Generate executive summary and compute overall recommendation strictly in code."""
        start_time = time.time()
        input_data = "".join(f"{c.type.value}:{c.risk.value if c.risk else 'NONE'}:{c.reason}" for c in state.clauses)
        input_hash = hashlib.sha256(input_data.encode("utf-8")).hexdigest()

        # Hard requirement: overall recommendation is computed in code, not by LLM (§4)
        overall_risk, recommendation = compute_overall_recommendation(state.clauses)

        # Prepare user payload for LLM summary
        clause_items = []
        for c in state.clauses:
            clause_items.append(
                f"- Clause: {c.type.value}\n"
                f"  Found: {c.found}\n"
                f"  Risk: {c.risk.value if c.risk else 'UNKNOWN'}\n"
                f"  Reason: {c.reason or 'None'}"
            )
        user_content = "Evaluated Clauses:\n" + "\n".join(clause_items)

        chat_history = ChatHistory()
        chat_history.add_system_message(self.system_prompt)
        chat_history.add_user_message(user_content)

        settings = OpenAIChatPromptExecutionSettings(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        validation_result = "passed"
        final_summary = ""
        raw_response = ""

        try:
            responses = await self.chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=settings,
            )
            raw_response = responses[0].content or ""
            final_summary = validate_summarizer_output(raw_response, state.clauses)

        except (OutputValidationError, Exception) as err:
            logger.warning("Summarizer first attempt failed validation: %s. Retrying once.", err)
            try:
                retry_history = ChatHistory()
                retry_history.add_system_message(self.system_prompt)
                retry_history.add_user_message(user_content)
                retry_history.add_assistant_message(raw_response)
                retry_history.add_user_message(
                    f"Your previous summary failed validation: {err}. "
                    "Write under 200 words, mention every flagged clause by name, and list REJECT and REVIEW first."
                )

                retry_responses = await self.chat_service.get_chat_message_contents(
                    chat_history=retry_history,
                    settings=settings,
                )
                retry_raw = retry_responses[0].content or ""
                final_summary = validate_summarizer_output(retry_raw, state.clauses)
                validation_result = "retried_and_passed"

            except (OutputValidationError, Exception) as retry_err:
                logger.error("Summarizer retry failed: %s. Using deterministic template fallback.", retry_err)
                final_summary = self._deterministic_template_fallback(state.clauses)
                validation_result = "fallback_used"

        latency = time.time() - start_time
        token_info = token_tracker.get_usage(self.agent_name)

        audit_entry = {
            "agent": self.agent_name,
            "input_hash": input_hash,
            "output": {
                "summary": final_summary,
                "overall_risk": overall_risk.value,
                "recommendation": recommendation.value,
            },
            "validation_result": validation_result,
            "tokens": token_info,
            "latency": round(latency, 4),
        }

        return final_summary, overall_risk, recommendation, audit_entry
