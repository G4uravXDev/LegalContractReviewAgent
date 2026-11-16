"""Clause Extractor Agent.

Extracts clauses of types: liability, termination, payment, confidentiality.
Enforces verbatim text copying, anti-hallucination substring checks,
retry on validation errors, and regex heading fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from config.agent_config import build_agent_kernel, get_agent_config, token_tracker
from guardrails.output_guard import OutputValidationError, validate_extractor_output
from state import Clause, ClauseType, ContractState

logger = logging.getLogger("contract_review.agents.extractor")


class ClauseExtractorAgent:
    """Agent responsible for identifying and extracting verbatim contract clauses."""

    def __init__(self, mock_mode: bool = False):
        self.agent_name = "extractor"
        self.mock_mode = mock_mode
        self.config = get_agent_config(self.agent_name)
        self.kernel, self.chat_service = build_agent_kernel(self.agent_name, mock_mode=mock_mode)
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "extractor.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _regex_heading_fallback(self, raw_text: str) -> List[Clause]:
        """Deterministic regex heading fallback when LLM output fails validation."""
        logger.warning("Invoking regex heading fallback for clause extraction.")
        clauses: List[Clause] = []

        heading_patterns = {
            ClauseType.LIABILITY: [
                r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Term\b|Termination\b|Payment\b|Fees\b|Confidentiality\b|Miscellaneous\b|Governing\b))|\Z)",
            ],
            ClauseType.TERMINATION: [
                r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Term and Termination|Termination)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability\b|Payment\b|Fees\b|Confidentiality\b|Miscellaneous\b|Governing\b))|\Z)",
            ],
            ClauseType.PAYMENT: [
                r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Fees and Payment Terms|Payment Terms|Payment|Fees)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability\b|Term\b|Termination\b|Confidentiality\b|Miscellaneous\b|Governing\b))|\Z)",
            ],
            ClauseType.CONFIDENTIALITY: [
                r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Confidentiality and Non-Disclosure|Confidentiality and Proprietary|Confidentiality|Non-Disclosure)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability\b|Term\b|Termination\b|Payment\b|Fees\b|Miscellaneous\b|Governing\b))|\Z)",
            ],
        }

        for c_type, patterns in heading_patterns.items():
            matched = False
            for pat in patterns:
                m = re.search(pat, raw_text, re.IGNORECASE)
                if m:
                    extracted_text = m.group(0).strip()
                    clauses.append(Clause(type=c_type, found=True, text=extracted_text))
                    matched = True
                    break
            if not matched:
                clauses.append(Clause(type=c_type, found=False, text=""))

        return clauses

    async def execute(self, state: ContractState) -> Tuple[List[Clause], Dict[str, Any]]:
        """Extract clauses from state.raw_text and return validated clauses + audit record."""
        start_time = time.time()
        input_data = state.raw_text
        input_hash = hashlib.sha256(input_data.encode("utf-8")).hexdigest()

        chat_history = ChatHistory()
        chat_history.add_system_message(self.system_prompt)
        chat_history.add_user_message(f"Contract Text to Review:\n{input_data}")

        settings = OpenAIChatPromptExecutionSettings(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"} if not self.mock_mode else None,
        )

        validation_result = "passed"
        extracted_clauses: List[Clause] = []
        raw_response_content = ""

        try:
            # First attempt
            responses = await self.chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=settings,
            )
            raw_response_content = responses[0].content or ""
            extracted_clauses = validate_extractor_output(raw_response_content, state.raw_text)

        except (OutputValidationError, Exception) as err:
            logger.warning("Extractor first attempt failed validation: %s. Retrying once.", err)
            try:
                # Retry with error feedback
                retry_history = ChatHistory()
                retry_history.add_system_message(self.system_prompt)
                retry_history.add_user_message(f"Contract Text to Review:\n{input_data}")
                retry_history.add_assistant_message(raw_response_content)
                retry_history.add_user_message(
                    f"Your previous output failed validation: {err}. "
                    "Extract ONLY verbatim substrings and return valid JSON."
                )

                retry_responses = await self.chat_service.get_chat_message_contents(
                    chat_history=retry_history,
                    settings=settings,
                )
                raw_response_content = retry_responses[0].content or ""
                extracted_clauses = validate_extractor_output(raw_response_content, state.raw_text)
                validation_result = "retried_and_passed"

            except (OutputValidationError, Exception) as retry_err:
                logger.error("Extractor retry also failed: %s. Falling back to regex extraction.", retry_err)
                extracted_clauses = self._regex_heading_fallback(state.raw_text)
                validation_result = "fallback_used"

        latency = time.time() - start_time
        token_info = token_tracker.get_usage(self.agent_name)

        audit_entry = {
            "agent": self.agent_name,
            "input_hash": input_hash,
            "output": [c.model_dump() for c in extracted_clauses],
            "validation_result": validation_result,
            "tokens": token_info,
            "latency": round(latency, 4),
        }

        return extracted_clauses, audit_entry
