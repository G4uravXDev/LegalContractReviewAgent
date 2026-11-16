"""Configuration, key management, and Semantic Kernel factory for all agents."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

logger = logging.getLogger("contract_review.config")

# Load environment variables from .env file
load_dotenv()


def mask_key(key: Optional[str]) -> str:
    """Mask sensitive API keys for safe logging."""
    if not key:
        return "[NOT_SET]"
    if len(key) <= 8:
        return "***"
    return f"{key[:3]}...{key[-4:]}"


@dataclass
class AgentConfig:
    """Per-agent execution configuration."""
    name: str
    api_key: Optional[str]
    model_id: str
    temperature: float
    max_tokens: int
    timeout: float = 60.0
    max_retries: int = 2
    base_url: Optional[str] = None

    def get_masked_key(self) -> str:
        return mask_key(self.api_key)


class TokenUsageTracker:
    """Tracks token consumption per agent."""
    def __init__(self):
        self._usage: Dict[str, Dict[str, int]] = {}

    def record_usage(self, agent_name: str, prompt_tokens: int, completion_tokens: int) -> None:
        if agent_name not in self._usage:
            self._usage[agent_name] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._usage[agent_name]["prompt_tokens"] += prompt_tokens
        self._usage[agent_name]["completion_tokens"] += completion_tokens
        self._usage[agent_name]["total_tokens"] += prompt_tokens + completion_tokens
        logger.info(
            "Agent [%s] token usage: +%d prompt, +%d completion (Agent total: %d)",
            agent_name, prompt_tokens, completion_tokens, self._usage[agent_name]["total_tokens"]
        )

    def get_usage(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        if agent_name:
            return self._usage.get(agent_name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        total = sum(v["total_tokens"] for v in self._usage.values())
        return {"per_agent": self._usage, "grand_total_tokens": total}


# Global usage tracker instance
token_tracker = TokenUsageTracker()


class DeterministicMockChatService(ChatCompletionClientBase):
    """Deterministic fake LLM service for testing and mock mode without API keys."""

    agent_name: str = "mock_agent"

    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> List[ChatMessageContent]:
        """Generate deterministic responses tailored to each agent role."""
        last_msg = ""
        for msg in chat_history.messages:
            if msg.role == AuthorRole.USER:
                last_msg = msg.content or ""

        # Rough token approximation
        prompt_tokens = max(10, len(last_msg) // 4)
        response_text = self._generate_mock_response(last_msg)
        completion_tokens = max(10, len(response_text) // 4)

        token_tracker.record_usage(self.agent_name, prompt_tokens, completion_tokens)

        return [
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=response_text,
                metadata={
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                },
            )
        ]

    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ):
        contents = await self.get_chat_message_contents(chat_history, settings, **kwargs)
        for c in contents:
            yield c

    def _generate_mock_response(self, prompt: str) -> str:
        """Route to appropriate deterministic mock generator."""
        agent = self.agent_name.lower()
        if "extractor" in agent:
            return self._mock_extractor(prompt)
        elif "flagger" in agent:
            return self._mock_flagger(prompt)
        elif "summarizer" in agent:
            return self._mock_summarizer(prompt)
        elif "retriever" in agent:
            return self._mock_retriever(prompt)
        return '{"status": "ok"}'

    def _mock_extractor(self, prompt: str) -> str:
        """Deterministic extractor: locates standard clause sections verbatim from contract."""
        match = re.search(r"<contract>\s*([\s\S]*?)\s*</contract>", prompt)
        contract = match.group(1) if match else prompt

        clauses = []

        pat_liab = r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Term\b|Termination\b|Payment\b|Fees\b|Confidentiality\b|Miscellaneous\b|Governing\b))|\Z)"
        pat_term = r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Term and Termination|Termination)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability\b|Payment\b|Fees\b|Confidentiality\b|Miscellaneous\b|Governing\b))|\Z)"
        pat_pay = r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Fees and Payment Terms|Payment Terms|Payment|Fees)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability\b|Term\b|Termination\b|Confidentiality\b|Miscellaneous\b|Governing\b))|\Z)"
        pat_conf = r"(?:^|\n)\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Confidentiality and Non-Disclosure|Confidentiality and Proprietary|Confidentiality|Non-Disclosure)\b[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:(?:\d+[\.\:]?|Section\s*\d+[\.\:]?)\s+)?(?:Limitation of Liability|Liability\b|Term\b|Termination\b|Payment\b|Fees\b|Miscellaneous\b|Governing\b))|\Z)"

        # 1. Liability
        m_liab = re.search(pat_liab, contract, re.IGNORECASE)
        if m_liab:
            clauses.append({"type": "liability", "found": True, "text": m_liab.group(0).strip()})
        else:
            clauses.append({"type": "liability", "found": False, "text": ""})

        # 2. Termination
        m_term = re.search(pat_term, contract, re.IGNORECASE)
        if m_term:
            clauses.append({"type": "termination", "found": True, "text": m_term.group(0).strip()})
        else:
            clauses.append({"type": "termination", "found": False, "text": ""})

        # 3. Payment
        m_pay = re.search(pat_pay, contract, re.IGNORECASE)
        if m_pay:
            clauses.append({"type": "payment", "found": True, "text": m_pay.group(0).strip()})
        else:
            clauses.append({"type": "payment", "found": False, "text": ""})

        # 4. Confidentiality
        m_conf = re.search(pat_conf, contract, re.IGNORECASE)
        if m_conf:
            clauses.append({"type": "confidentiality", "found": True, "text": m_conf.group(0).strip()})
        else:
            clauses.append({"type": "confidentiality", "found": False, "text": ""})

        import json
        return json.dumps({"clauses": clauses})

    def _mock_flagger(self, prompt: str) -> str:
        """Deterministic risk flagger based on policy rules."""
        import json

        # Crucial: Extract clause text strictly from inside <contract_clause> tags
        m = re.search(r"<contract_clause>\s*([\s\S]*?)\s*</contract_clause>", prompt, re.IGNORECASE)
        clause_text = m.group(1).lower() if m else prompt.lower()

        # Check for unacceptable conditions in the actual clause text (REJECT)
        if "unlimited liability" in clause_text or "without cap" in clause_text or "uncapped" in clause_text:
            return json.dumps({
                "risk": "REJECT",
                "reason": "Matches unacceptable variant: contract specifies unlimited liability without monetary cap."
            })
        if "no termination for convenience" in clause_text:
            return json.dumps({
                "risk": "REJECT",
                "reason": "Matches unacceptable variant: agreement lacks standard termination for convenience."
            })
        if "net 90" in clause_text or "net 120" in clause_text:
            return json.dumps({
                "risk": "REJECT",
                "reason": "Matches unacceptable variant: payment window exceeds acceptable commercial limits."
            })

        # Check for review/deviation conditions in the actual clause text (REVIEW)
        if "90 days" in clause_text or "ninety (90) days" in clause_text:
            return json.dumps({
                "risk": "REVIEW",
                "reason": "Notice period of 90 days deviates from the preferred 30-day and acceptable 60-day standard."
            })
        if "net 60" in clause_text or "sixty (60) days" in clause_text:
            return json.dumps({
                "risk": "REVIEW",
                "reason": "Payment term of Net 60 days exceeds the standard Net 30 preferred and Net 45 acceptable position."
            })
        if "24 months" in clause_text or "two times" in clause_text or "2x" in clause_text:
            # 24 months is in the acceptable variant, so it is ACCEPTABLE
            return json.dumps({
                "risk": "ACCEPTABLE",
                "reason": "Liability cap of 24 months fees matches the firm's acceptable variant standard."
            })

        # Standard acceptable conditions
        return json.dumps({
            "risk": "ACCEPTABLE",
            "reason": "Clause terms align with the firm's preferred standard policy position."
        })

    def _mock_summarizer(self, prompt: str) -> str:
        """Deterministic summarizer mentioning every flagged clause by name (<200 words)."""
        # Parse clause items from the prompt
        clause_matches = re.findall(
            r"Clause:\s*(\w+)\s*\n\s*Found:\s*(\w+)\s*\n\s*Risk:\s*(\w+)\s*\n\s*Reason:\s*([^\n]+)",
            prompt,
            re.IGNORECASE,
        )

        rejects = []
        reviews = []
        acceptable = []

        for c_type, found_str, risk_str, reason_str in clause_matches:
            c_name = c_type.lower()
            risk_val = risk_str.upper()
            reason_clean = reason_str.strip()
            if found_str.lower() == "false" or risk_val == "REVIEW":
                reviews.append((c_name, reason_clean or "Deviates from standard position."))
            elif risk_val == "REJECT":
                rejects.append((c_name, reason_clean or "Unacceptable deviation from policy."))
            elif risk_val == "ACCEPTABLE":
                acceptable.append(c_name)

        # Fallback if pattern didn't match (e.g. ad-hoc user query)
        if not clause_matches:
            for c_name in ["liability", "termination", "payment", "confidentiality"]:
                if c_name in prompt.lower():
                    acceptable.append(c_name)

        lines = ["Contract Review Executive Summary for Reviewer:"]

        if rejects:
            lines.append("Rejections:")
            for c_name, r_reason in rejects:
                lines.append(f"- {c_name.capitalize()} (REJECT): {r_reason}")

        if reviews:
            lines.append("Provisions Requiring Review or Negotiation:")
            for c_name, r_reason in reviews:
                lines.append(f"- {c_name.capitalize()} (REVIEW): {r_reason}")

        if not rejects and not reviews:
            lines.append("All analyzed provisions fully adhere to the standard policy positions.")

        if acceptable:
            acc_str = ", ".join(c.capitalize() for c in acceptable)
            lines.append(f"Acceptable clauses: {acc_str} comply with standard firm positions.")

        return "\n".join(lines)

    def _mock_retriever(self, prompt: str) -> str:
        return "liability cap notice payment term"


def get_agent_config(agent_name: str) -> AgentConfig:
    """Retrieve configuration for a specific agent from environment variables."""
    key_env_var = f"{agent_name.upper()}_API_KEY"
    model_env_var = f"{agent_name.upper()}_MODEL"
    base_url_env_var = f"{agent_name.upper()}_BASE_URL"

    api_key = os.getenv(key_env_var)
    model_id = os.getenv(model_env_var, "gpt-4o-mini")
    base_url = os.getenv(base_url_env_var)

    # Defaults per agent role specified in §6
    temperatures = {
        "extractor": 0.0,
        "flagger": 0.0,
        "summarizer": 0.3,
        "retriever": 0.0,
    }
    max_tokens_map = {
        "extractor": 2000,
        "flagger": 500,
        "summarizer": 800,
        "retriever": 500,
    }

    temp = temperatures.get(agent_name.lower(), 0.0)
    max_tok = max_tokens_map.get(agent_name.lower(), 1000)

    return AgentConfig(
        name=agent_name.lower(),
        api_key=api_key,
        model_id=model_id,
        temperature=temp,
        max_tokens=max_tok,
        base_url=base_url,
    )


def validate_all_keys(mock_mode: bool = False) -> None:
    """Fail fast at startup if any required API key is missing (§6 requirement)."""
    if mock_mode:
        logger.info("Running in mock mode. API keys are not required.")
        return

    required_agents = ["extractor", "retriever", "flagger", "summarizer"]
    missing = []

    for agent in required_agents:
        cfg = get_agent_config(agent)
        if not cfg.api_key or not cfg.api_key.strip():
            missing.append(f"{agent.upper()}_API_KEY")

    if missing:
        err_msg = (
            f"Missing required API key(s): {', '.join(missing)}. "
            "Set them in your .env file or use --mock mode for deterministic local execution."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def build_agent_kernel(
    agent_name: str,
    mock_mode: bool = False,
) -> Tuple[Kernel, ChatCompletionClientBase]:
    """Create a dedicated Semantic Kernel and chat completion service for one agent.
    
    Hard requirement: One Kernel and one chat/embedding service per agent.
    Each agent receives only its own key.
    """
    cfg = get_agent_config(agent_name)
    kernel = Kernel()

    service_id = f"{agent_name.lower()}_chat"

    if mock_mode:
        logger.info("Initializing agent [%s] with DeterministicMockChatService.", agent_name)
        service = DeterministicMockChatService(
            ai_model_id=f"mock-{cfg.model_id}",
            service_id=service_id,
            agent_name=agent_name,
        )
    else:
        if not cfg.api_key:
            raise ValueError(f"Missing API key for agent '{agent_name}'. Cannot build real Kernel service.")
        logger.info(
            "Initializing agent [%s] with OpenAIChatCompletion (model: %s, key: %s, base_url: %s).",
            agent_name, cfg.model_id, cfg.get_masked_key(), cfg.base_url or "default(OpenAI)"
        )
        if cfg.base_url and cfg.base_url.strip():
            from openai import AsyncOpenAI
            custom_client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url.strip(), timeout=cfg.timeout)
            service = OpenAIChatCompletion(
                ai_model_id=cfg.model_id,
                service_id=service_id,
                async_client=custom_client,
            )
        else:
            service = OpenAIChatCompletion(
                ai_model_id=cfg.model_id,
                api_key=cfg.api_key,
                service_id=service_id,
            )

    kernel.add_service(service)
    return kernel, service
